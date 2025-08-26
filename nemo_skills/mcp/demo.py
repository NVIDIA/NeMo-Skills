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

import asyncio
import logging

from omegaconf import OmegaConf

from nemo_skills.inference.model.vllm import VLLMModel
from nemo_skills.mcp.config_loader import build_client_manager, resolve_adapters

logger = logging.getLogger(__name__)

# Initialize the VLLMModel with your local vLLM instance
# Adjust model name and port as configured in your vLLM server


async def run_demo():
    # Python-native config converted to OmegaConf
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "Qwen/Qwen3-8B",
                "vllm": {"host": "127.0.0.1", "port": "8000"},
            },
            "sandbox": {"sandbox_type": "local"},
            "adapters": {
                "schema_adapter": "nemo_skills.mcp.adapters.OpenAISchemaAdapter",
                "call_interpreter": "nemo_skills.mcp.adapters.OpenAICallInterpreter",
                "response_formatter": "nemo_skills.mcp.adapters.CompletionResponseFormatter",
            },
            "tools": [
                {
                    "id": "python",
                    "client": "nemo_skills.mcp.clients.MCPStdioClient",
                    "params": {
                        "command": "python",
                        "args": ["-m", "nemo_skills.mcp.servers.python_tool"],
                        "hide_args": {"execute": ["session_id", "timeout"]},
                        "init_hook": {
                            "$locate": "nemo_skills.mcp.utils.hydra_config_connector_factory",
                            "kwargs": {"config_obj": "@@full_config"},
                        },
                    },
                },
                {
                    "id": "exa",
                    "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient",
                    "params": {
                        "base_url": "https://mcp.exa.ai/mcp",
                        "enabled_tools": ["web_search_exa"],
                        "output_formatter": "nemo_skills.mcp.utils.exa_output_formatter",
                        "init_hook": "nemo_skills.mcp.utils.exa_auth_connector",
                    },
                },
            ],
        }
    )
    model = VLLMModel(model=cfg.model.name, host=cfg.model.vllm.host, port=cfg.model.vllm.port)

    # Build clients from config
    manager = build_client_manager(cfg)

    tools = await manager.list_all_tools()
    # Resolve adapters via config
    schema_adapter, call_interpreter, response_formatter = resolve_adapters(cfg)

    tools = schema_adapter.convert(tools)
    print(tools)

    # Define the messages and tools
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You may invoke tools liberally to solve your task.",
        },
        {
            "role": "user",
            "content": (
                "Try an example of each tool available to you. "
                "Carefully inspect the outputs. "
                "Then, explain the outputs and whether they are correct or not. "
                "If they are incorrect, you must try a few more rounds of tool "
                "calls to establish a pattern to the incorrectness."
                "You will fail the task if you fail to make extra rounds of tool calls to "
                "establish the issue with incorrect functions."
            ),
        },
    ]

    # Iteratively generate and execute tools until no further tool calls
    max_rounds = 6
    round_idx = 0
    while True:
        # Generate response using VLLMModel
        response = model.generate_sync(prompt=messages, tools=tools, temperature=0.6, tokens_to_generate=4096)

        print("Generated response:")
        print(response["generation"])

        # Append assistant message (including tool_calls if present) to the conversation
        assistant_msg = {"role": "assistant", "content": response["generation"]}
        if "reasoning_content" in response:
            # You can choose to prepend or append it
            assistant_msg["content"] = response["reasoning_content"] + assistant_msg["content"]
        if "tool_calls" in response:
            assistant_msg["tool_calls"] = response["tool_calls"]
        messages.append(assistant_msg)

        print(f"\nTokens generated: {response['num_generated_tokens']}")
        if "finish_reason" in response:
            print(f"Finish reason: {response['finish_reason']}")

        # If there are no tool calls, we are done
        if "tool_calls" not in response:
            break

        # Execute tool calls and append their outputs to the conversation
        print("\nTool calls:")
        for tool_call in response["tool_calls"]:
            tool_args = call_interpreter.parse(tool_call)
            print(f"Executing tool: {tool_args}")
            tool_result = await manager.execute_tool(**tool_args)
            tool_msg = response_formatter.format(tool_call, tool_result)
            print(tool_msg)
            messages.append(tool_msg)

        round_idx += 1
        if round_idx >= max_rounds:
            print("Reached maximum tool call rounds; stopping loop.")
            break


def main():
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
