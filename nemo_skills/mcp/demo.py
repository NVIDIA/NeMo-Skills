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
from nemo_skills.inference.model.vllm import VLLMModel
from nemo_skills.mcp.clients import MCPHttpClient, MCPClientManager
from nemo_skills.mcp.adapters import registry

# Initialize the VLLMModel with your local vLLM instance
# Adjust model name and port as configured in your vLLM server
async def run_demo():
    model = VLLMModel(
        model="Qwen/Qwen3-8B",
        host="127.0.0.1",
        port="8000"
    )
    model_type = 'qwen'
    math_client = MCPHttpClient("http://localhost:8001")
    string_client = MCPHttpClient("http://localhost:8002")

    manager = MCPClientManager()
    manager.register("math", math_client)
    manager.register("string", string_client)

    tools = await manager.list_all_tools()
    tools = registry.get_schema(model_type).convert(tools)
    print(tools)

    # Define the messages and tools
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You may invoke tools when required."},
        # {"role": "user", "content": "What is the weather in London and Paris right now?"}
        {"role": "user", "content": "Try an example of each tool available to you. Carefully inspect the outputs. Then, explain the outputs and whether they are correct or not."}
    ]

    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_current_weather",
    #             "description": "Get the current weather for a specific location",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "location": {"type": "string", "description": "The city and country to get weather for"}
    #                 },
    #                 "required": ["location"]
    #             }
    #         }
    #     }
    # ]

    # Iteratively generate and execute tools until no further tool calls
    max_rounds = 6
    round_idx = 0
    while True:
        # Generate response using VLLMModel
        response = model.generate_sync(
            prompt=messages,
            tools=tools,
            temperature=0.0,
            tokens_to_generate=512
        )

        print("Generated response:")
        print(response["generation"])

        # Append assistant message (including tool_calls if present) to the conversation
        assistant_msg = {"role": "assistant", "content": response["generation"]}
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
            tool_args = registry.get_interpreter(model_type).parse(tool_call)
            tool_result = await manager.execute_tool(**tool_args)
            tool_msg = registry.get_response_formatter(model_type).format(tool_call, tool_result)
            print(tool_msg)
            messages.append(tool_msg)

        round_idx += 1
        if round_idx >= max_rounds:
            print("Reached maximum tool call rounds; stopping loop.")
            break


if __name__ == "__main__":
    asyncio.run(run_demo())
