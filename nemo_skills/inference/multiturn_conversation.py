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
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from tqdm import tqdm

from nemo_skills.inference.model import get_model
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ServerConfig:
    """Configuration for server connection."""

    server_type: str = "vllm"  # Type of server (vllm, sglang, trtllm, etc.)
    model: str = ""  # Model path or name (set from parent config)
    host: str = "127.0.0.1"  # Server host (resolved at runtime)
    port: str = "5000"  # Server port (resolved at runtime - must be string)
    ssh_server: Optional[str] = None
    ssh_key_path: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3


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
    server_a_address: str  # Address of model A server (ip:port)
    server_b_address: str  # Address of model B server (ip:port)
    num_turns: int = 4  # Number of conversation turns
    model_a_name: str = "model_a"  # Name to use for model A in output
    model_b_name: str = "model_b"  # Name to use for model B in output

    # Inference parameters for model A
    inference_a: MultiTurnInferenceConfig = field(default_factory=MultiTurnInferenceConfig)
    # Inference parameters for model B
    inference_b: MultiTurnInferenceConfig = field(default_factory=MultiTurnInferenceConfig)

    # Server configuration for model A
    server_a: ServerConfig = field(default_factory=ServerConfig)
    # Server configuration for model B
    server_b: ServerConfig = field(default_factory=ServerConfig)

    # Control settings
    skip_filled: bool = True  # Skip conversations that are already complete
    max_concurrent: int = 10  # Maximum concurrent API calls
    starting_prompt_key: str = "problem"  # Key in input JSONL containing the initial prompt


@hydra.main(version_base=None, config_path=None)
def main(cfg: MultiTurnConfig):
    cfg = MultiTurnConfig(_init_nested=True, **cfg)
    LOG.info("Multi-turn conversation configuration:")
    LOG.info(f"  Input file: {cfg.input_file}")
    LOG.info(f"  Output file: {cfg.output_file}")
    LOG.info(f"  Model A: {cfg.model_a_name} @ {cfg.server_a_address}")
    LOG.info(f"  Model B: {cfg.model_b_name} @ {cfg.server_b_address}")
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

    # Create model clients
    LOG.info("Connecting to model servers...")

    # Parse server addresses
    server_a_host, server_a_port = cfg.server_a_address.rsplit(":", 1)
    server_b_host, server_b_port = cfg.server_b_address.rsplit(":", 1)

    # Configure server params
    cfg.server_a.model = cfg.model_a_name
    cfg.server_a.host = server_a_host
    cfg.server_a.port = server_a_port  # Keep as string
    cfg.server_b.model = cfg.model_b_name
    cfg.server_b.host = server_b_host
    cfg.server_b.port = server_b_port  # Keep as string

    # Convert server configs to dict for get_model
    # Filter out empty strings but keep all other values
    server_a_dict = {k: v for k, v in cfg.server_a.__dict__.items() if v != ""}
    server_b_dict = {k: v for k, v in cfg.server_b.__dict__.items() if v != ""}

    # Create model instances (do NOT pass inference params here - those go to API calls)
    model_a = get_model(**server_a_dict)
    model_b = get_model(**server_b_dict)

    LOG.info("Successfully connected to both model servers")

    # Run conversation generation
    results = asyncio.run(
        generate_conversations(
            input_data=input_data,
            existing_conversations=existing_conversations,
            model_a=model_a,
            model_b=model_b,
            num_turns=cfg.num_turns,
            model_a_name=cfg.model_a_name,
            model_b_name=cfg.model_b_name,
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

                # Track current message for next turn
                current_message = initial_prompt

                # Generate turns
                for turn_idx in range(num_turns):
                    is_model_a = turn_idx % 2 == 0
                    model = model_a if is_model_a else model_b
                    speaker = "model_a" if is_model_a else "model_b"
                    model_name = model_a_name if is_model_a else model_b_name

                    # Create messages for the LLM
                    # For first turn, just use the initial prompt
                    # For subsequent turns, build conversation history
                    if turn_idx == 0:
                        messages = [{"role": "user", "content": current_message}]
                    else:
                        # Build conversation history
                        messages = [{"role": "user", "content": initial_prompt}]
                        for prev_turn in conversation["turns"]:
                            # Alternate roles based on who spoke
                            if prev_turn["speaker"] == "model_a":
                                role = "assistant" if is_model_a else "user"
                            else:
                                role = "user" if is_model_a else "assistant"
                            messages.append({"role": role, "content": prev_turn["content"]})

                    # Generate response
                    try:
                        # Get inference params for current model
                        inference_params = inference_a.__dict__ if is_model_a else inference_b.__dict__
                        response_data = await model.call_async(messages, **inference_params)
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

                        # Update current message for next turn
                        current_message = response_content

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
