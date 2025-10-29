# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-turn conversation generation between two models.

This module orchestrates conversations between two language models by:
1. Reading initial prompts from a JSONL file
2. Alternating requests between two model servers
3. Writing complete conversations in chat format to output JSONL

Output format:
{
    "conversation_id": 0,
    "initial_prompt": "Discuss the implications of AI",
    "turns": [
        {"speaker": "model_a", "model": "llama-3-8b", "content": "I think AI will..."},
        {"speaker": "model_b", "model": "llama-3-70b", "content": "I disagree because..."},
        {"speaker": "model_a", "model": "llama-3-8b", "content": "That's a good point..."},
        ...
    ],
    "metadata": {
        "num_turns": 3,
        "model_a": "llama-3-8b",
        "model_b": "llama-3-70b"
    }
}
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import hydra
from tqdm import tqdm

from nemo_skills.inference.model import get_model
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def wait_for_servers_ready(
    servers: Dict[str, str],
    timeout: int = 600,
    poll_interval: int = 5,
) -> Dict[str, Dict]:
    """Wait for model servers to be ready before proceeding.

    Args:
        servers: Dict mapping server names to /v1/models endpoint URLs
        timeout: Maximum time to wait in seconds (default: 10 minutes)
        poll_interval: Time between health check attempts in seconds

    Returns:
        Dict mapping server names to their model info from /v1/models endpoint

    Raises:
        TimeoutError: If servers are not ready within timeout period
    """
    LOG.info(f"Waiting for {len(servers)} server(s) to be ready (timeout: {timeout}s)...")

    deadline = time.time() + timeout
    ready_servers = {}

    while time.time() < deadline:
        for name, url in servers.items():
            if name in ready_servers:
                continue
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    model_info = resp.json()
                    LOG.info(f"✓ {name} is ready!")
                    ready_servers[name] = model_info
            except Exception:
                pass  # Server not ready yet

        if len(ready_servers) == len(servers):
            LOG.info("All servers are ready!")
            return ready_servers

        time.sleep(poll_interval)

    missing = set(servers.keys()) - set(ready_servers.keys())
    raise TimeoutError(f"Servers not ready after {timeout}s. Still waiting for: {', '.join(missing)}")


@nested_dataclass(kw_only=True)
class MultiTurnInferenceConfig:
    """Configuration for multi-turn conversation inference."""

    temperature: float = 0.7  # Higher temperature for more diverse conversations
    top_k: int = -1
    top_p: float = 0.95
    tokens_to_generate: int = 2048
    timeout: int = 14400  # Timeout for each LLM call in seconds
    extra_body: dict = field(default_factory=dict)


@nested_dataclass(kw_only=True)
class MultiTurnConfig:
    """Configuration for multi-turn conversation generation."""

    input_file: str  # JSONL file with initial prompts (each line should have a "problem" or "question" field)
    output_file: str  # Where to save the conversations

    # Multi-model configuration
    server_addresses: List[str] = field(default_factory=list)  # List of server addresses for N models
    model_names: List[str] = field(default_factory=list)  # List of model names for N models

    num_turns: int = 4  # Number of conversation turns

    # Inference parameters for each model
    inference_a: MultiTurnInferenceConfig = field(default_factory=MultiTurnInferenceConfig)
    inference_b: MultiTurnInferenceConfig = field(default_factory=MultiTurnInferenceConfig)

    # Control settings
    skip_filled: bool = True  # Skip conversations that are already complete
    max_concurrent: int = 10  # Maximum concurrent API calls
    starting_prompt_key: str = "problem"  # Key in input JSONL containing the initial prompt

    # Server configuration (optional, for advanced use cases)
    server_type: str = "vllm"

    def __post_init__(self):
        """Validate configuration."""
        if not self.server_addresses:
            raise ValueError("Must specify server_addresses for multi-turn conversation")

        if len(self.server_addresses) < 2:
            raise ValueError(f"Multi-turn conversation requires at least 2 models, got {len(self.server_addresses)}")

        # Ensure model_names matches server_addresses
        if not self.model_names:
            self.model_names = [f"model_{i}" for i in range(len(self.server_addresses))]
        elif len(self.model_names) != len(self.server_addresses):
            raise ValueError(
                f"Number of model_names ({len(self.model_names)}) must match "
                f"number of server_addresses ({len(self.server_addresses)})"
            )


@hydra.main(version_base=None, config_path=None)
def main(cfg: MultiTurnConfig):
    cfg = MultiTurnConfig(_init_nested=True, **cfg)
    LOG.info("Multi-turn conversation configuration:")
    LOG.info(f"  Input file: {cfg.input_file}")
    LOG.info(f"  Output file: {cfg.output_file}")
    LOG.info(f"  Number of models: {len(cfg.server_addresses)}")
    for idx, (model_name, address) in enumerate(zip(cfg.model_names, cfg.server_addresses)):
        LOG.info(f"  Model {idx}: {model_name} @ {address}")
    LOG.info(f"  Number of turns: {cfg.num_turns}")

    # Read input prompts
    input_path = Path(cfg.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    LOG.info(f"Loaded {len(input_data)} initial prompts")

    # Check for existing output and determine what to process
    output_path = Path(cfg.output_file)
    existing_conversations = {}

    if output_path.exists() and cfg.skip_filled:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                conv = json.loads(line)
                conv_id = conv.get("conversation_id")
                if conv_id is not None:
                    existing_conversations[conv_id] = conv
        LOG.info(f"Found {len(existing_conversations)} existing conversations, will skip those")

    # Wait for all servers to be ready before proceeding
    servers_to_check = {}
    for idx, address in enumerate(cfg.server_addresses):
        host, port = address.rsplit(":", 1)
        servers_to_check[f"Server {idx}"] = f"http://{host}:{port}/v1/models"

    wait_for_servers_ready(servers_to_check)

    # Create model instances directly from server addresses
    model_clients = []
    for idx, (address, model_name) in enumerate(zip(cfg.server_addresses, cfg.model_names)):
        host, port = address.rsplit(":", 1)
        LOG.info(f"Creating client for Server {idx}: {model_name} @ {host}:{port}")

        server_config = {
            "server_type": cfg.server_type,
            "model": model_name,
            "host": host,
            "port": port,
        }

        model_client = get_model(**server_config)
        model_clients.append(model_client)

    # For now, we still use model_a and model_b for the conversation logic
    # (assumes 2 models - can be extended later)
    model_a = model_clients[0]
    model_b = model_clients[1]

    # Run conversation generation
    results = asyncio.run(
        generate_conversations(
            input_data=input_data,
            existing_conversations=existing_conversations,
            model_a=model_a,
            model_b=model_b,
            num_turns=cfg.num_turns,
            model_a_name=cfg.model_names[0],
            model_b_name=cfg.model_names[1] if len(cfg.model_names) > 1 else cfg.model_names[0],
            starting_prompt_key=cfg.starting_prompt_key,
            max_concurrent=cfg.max_concurrent,
            inference_a=cfg.inference_a,
            inference_b=cfg.inference_b,
        )
    )

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing data if skip_filled was used
    if cfg.skip_filled and existing_conversations:
        # Combine new and existing, preferring new results
        all_results = {r["conversation_id"]: r for r in results}
        for conv_id, conv in existing_conversations.items():
            if conv_id not in all_results:
                all_results[conv_id] = conv
        # Sort by conversation_id
        results = [all_results[i] for i in sorted(all_results.keys())]

    with open(output_path, "w", encoding="utf-8") as f:
        for conversation in results:
            f.write(json.dumps(conversation) + "\n")

    LOG.info(f"Wrote {len(results)} conversations to {cfg.output_file}")


async def generate_conversations(
    input_data: List[Dict[str, Any]],
    existing_conversations: Dict[int, Dict],
    model_a: Any,
    model_b: Any,
    num_turns: int,
    model_a_name: str,
    model_b_name: str,
    starting_prompt_key: str,
    max_concurrent: int,
    inference_a: MultiTurnInferenceConfig,
    inference_b: MultiTurnInferenceConfig,
) -> List[Dict]:
    """Generate multi-turn conversations asynchronously.

    Args:
        input_data: List of input prompts
        existing_conversations: Already completed conversations to skip
        model_a: First model client
        model_b: Second model client
        num_turns: Number of turns per conversation
        model_a_name: Name for model A in output
        model_b_name: Name for model B in output
        starting_prompt_key: Key to extract initial prompt from input data
        max_concurrent: Maximum concurrent API calls
        inference_a: Inference config for model A
        inference_b: Inference config for model B

    Returns:
        List of conversation dictionaries
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_single_conversation(idx: int, prompt_data: Dict) -> Optional[Dict]:
        """Generate a single conversation."""
        # Skip if already exists
        if idx in existing_conversations:
            return None

        async with semaphore:
            try:
                # Extract initial prompt
                if starting_prompt_key not in prompt_data:
                    LOG.warning(f"Prompt {idx} missing '{starting_prompt_key}' field, skipping")
                    return None

                initial_prompt = prompt_data[starting_prompt_key]

                conversation = {
                    "conversation_id": idx,
                    "initial_prompt": initial_prompt,
                    "turns": [],
                    "metadata": {
                        "num_turns": num_turns,
                        "model_a": model_a_name,
                        "model_b": model_b_name,
                    },
                }

                # Generate turns
                for turn_idx in range(num_turns):
                    is_model_a = turn_idx % 2 == 0
                    model = model_a if is_model_a else model_b
                    speaker = "model_a" if is_model_a else "model_b"
                    model_name = model_a_name if is_model_a else model_b_name

                    # Create messages for the LLM with clear engagement structure
                    if turn_idx == 0:
                        # First turn: respond to the initial prompt
                        system_msg = "You are in a conversation. Keep your responses concise (2-3 sentences)."
                        messages = [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": initial_prompt},
                        ]
                    else:
                        # Subsequent turns: Focus on the immediate exchange
                        # Show just the last 2-3 turns to keep context clear
                        system_msg = (
                            f"You are in a conversation about: '{initial_prompt}'\n"
                            f"Keep your responses concise (2-3 sentences) and engage with what the other person just said."
                        )
                        messages = [{"role": "system", "content": system_msg}]

                        # Show recent history (last 4 turns or all if fewer)
                        recent_turns = (
                            conversation["turns"][-4:] if len(conversation["turns"]) > 4 else conversation["turns"]
                        )

                        for prev_turn in recent_turns:
                            # From current model's perspective:
                            # - Its own previous messages are "assistant"
                            # - The other model's messages are "user"
                            is_own_turn = prev_turn["speaker"] == speaker
                            role = "assistant" if is_own_turn else "user"
                            messages.append({"role": role, "content": prev_turn["content"]})

                        # Last message should always be from the other model (role="user")
                        # Add explicit engagement instruction
                        if messages[-1]["role"] == "user":
                            messages[-1]["content"] += (
                                "\n\n[Respond to the above. Engage with their specific points - "
                                "agree/disagree, ask a follow-up, challenge, or build on their idea.]"
                            )

                    # Generate response
                    try:
                        # Get inference params for current model
                        inference_params = inference_a.__dict__ if is_model_a else inference_b.__dict__
                        response_data = await model.generate_async(
                            prompt=messages, endpoint_type=EndpointType.chat, **inference_params
                        )
                        response_content = response_data.get("generation", "")

                        # Record turn
                        conversation["turns"].append(
                            {
                                "speaker": speaker,
                                "model": model_name,
                                "content": response_content,
                                "turn_idx": turn_idx,
                            }
                        )

                    except Exception as e:
                        LOG.error(f"Error generating turn {turn_idx} for conversation {idx}: {e}")
                        conversation["error"] = {
                            "turn": turn_idx,
                            "message": str(e),
                        }
                        break

                return conversation

            except Exception as e:
                LOG.error(f"Error processing conversation {idx}: {e}")
                return {
                    "conversation_id": idx,
                    "error": str(e),
                }

    # Generate all conversations
    tasks = [generate_single_conversation(idx, prompt_data) for idx, prompt_data in enumerate(input_data)]

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating conversations"):
        result = await task
        if result is not None:
            results.append(result)

    return results


class MultiTurnGenerationTask:
    """Task class for multi-turn conversation generation (required by pipeline)."""

    @classmethod
    def get_generation_default_args(cls) -> str:
        """Returns default arguments for multi-turn generation."""
        return ""

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """Returns the function to build server commands."""
        from nemo_skills.pipeline.utils import get_server_command

        return get_server_command


# Required by pipeline to identify this as a generation module
GENERATION_TASK_CLASS = MultiTurnGenerationTask


def run():
    setup_logging(disable_hydra_logs=False)
    help_message = get_help_message(MultiTurnConfig)
    if "--help" in sys.argv or "-h" in sys.argv:
        print(help_message)
    else:
        setup_logging()
        main()


if __name__ == "__main__":
    run()
