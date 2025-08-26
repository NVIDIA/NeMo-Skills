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
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from nemo_skills.inference.model.vllm import VLLMModel

# Adapter classes are resolved via locate at runtime
from nemo_skills.mcp.clients import MCPClientManager, MCPStdioClient, MCPStreamableHttpClient
from nemo_skills.mcp.utils import hydra_config_connector_factory, locate

logger = logging.getLogger(__name__)

# Initialize the VLLMModel with your local vLLM instance
# Adjust model name and port as configured in your vLLM server


@dataclass
class PythonToolMCPConfig:
    pass


@dataclass
class DemoConfig:
    model: str = "Qwen/Qwen3-8B"
    vllm_host: str = "127.0.0.1"
    vllm_port: str = "8000"
    python_tool: PythonToolMCPConfig = field(default_factory=PythonToolMCPConfig)
    # Class paths or instances; resolved via locate at runtime
    schema_adapter: str = "nemo_skills.mcp.adapters.OpenAISchemaAdapter"
    call_interpreter: str = "nemo_skills.mcp.adapters.OpenAICallInterpreter"
    response_formatter: str = "nemo_skills.mcp.adapters.QwenResponseFormatter"
    sandbox: dict = field(
        default_factory=lambda: {
            "sandbox_type": "local",
        }
    )


cs = ConfigStore.instance()
cs.store(name="base_mcp_demo_config", node=DemoConfig)


async def run_demo(cfg: DemoConfig):
    model = VLLMModel(model=cfg.model, host=cfg.vllm_host, port=cfg.vllm_port)
    # Build a Hydra config for the python MCP server and inject via connector
    # Accept whatever object Hydra parsed (dataclass/DictConfig) and convert to plain dict
    python_client = MCPStdioClient(
        "python",
        ["-m", "nemo_skills.mcp.servers.python_tool"],
        hide_args={"execute": ["session_id", "timeout"]},
        init_hook=hydra_config_connector_factory(cfg),
    )
    exa_client = MCPStreamableHttpClient(
        base_url="https://mcp.exa.ai/mcp",
        enabled_tools=["web_search_exa"],
        output_formatter=locate("nemo_skills.mcp.utils.exa_output_formatter"),
        init_hook=locate("nemo_skills.mcp.utils.exa_auth_connector"),
    )

    manager = MCPClientManager()
    manager.register("python", python_client)
    manager.register("exa", exa_client)

    tools = await manager.list_all_tools()
    # Resolve adapters via locate-only
    schema_adapter_obj = locate(cfg.schema_adapter)
    call_interpreter_obj = locate(cfg.call_interpreter)
    response_formatter_obj = locate(cfg.response_formatter)

    schema_adapter = schema_adapter_obj() if isinstance(schema_adapter_obj, type) else schema_adapter_obj
    call_interpreter = call_interpreter_obj() if isinstance(call_interpreter_obj, type) else call_interpreter_obj
    response_formatter = (
        response_formatter_obj() if isinstance(response_formatter_obj, type) else response_formatter_obj
    )

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


# @hydra.main(version_base=None, config_name="base_mcp_demo_config")
# def main(cfg: DemoConfig):
def main():
    cfg = DemoConfig(sandbox={"sandbox_type": "local"})
    asyncio.run(run_demo(cfg))


if __name__ == "__main__":
    main()
