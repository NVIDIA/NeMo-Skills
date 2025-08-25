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

import json
import logging
import textwrap
from typing import Dict, List

from litellm.types.utils import ChatCompletionMessageToolCall

from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.utils import get_logger_name

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


class ToolCallingWrapper:
    """
    Wrapper to handle tool calling.

    TODO(sanyamk): Supports only Chat Completions API for now.
    """

    def __init__(self, model: BaseModel, sandbox: Sandbox):
        self.model = model
        self.sandbox = sandbox

    async def _parse_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]):
        """Convert tool calls to conversation message item."""

        tool_calls = [
            {
                "type": tool_call.type,
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
            for tool_call in tool_calls
        ]

        return {"role": "assistant", "tool_calls": tool_calls}

    ## TODO(sanyamk): Support tool call IDs.
    async def _execute_tool_call(self, tool_name: str, tool_args):
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.decoder.JSONDecodeError as e:
                LOG.exception(e)

                return {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps({"error": "Tool execution failed, unable to parse arguments."}),
                }

        ## TODO(sanyamk): Replace tool handlers with MCP.
        if tool_name == "exa_websearch":
            tool_code = textwrap.dedent(
                f"""
                from exa_py import Exa
                import os

                exa = Exa(os.getenv("EXA_API_KEY"))

                result = exa.answer({repr(tool_args["query"])})

                print(result.answer)
            """
            )
        else:
            try:
                raise NotImplementedError
            except NotImplementedError as e:
                LOG.exception(e)

            return {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps({"error": "Tool not supported."}),
            }

        tool_output, _ = await self.sandbox.execute_code(generated_code=tool_code, language="python")

        if tool_output["process_status"] != "completed":
            try:
                raise ValueError(tool_output["stderr"])
            except ValueError as e:
                LOG.exception(e)

            return {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps({"error": tool_output["stderr"]}),
            }

        return {
            "role": "tool",
            "name": tool_name,
            "content": json.dumps({"result": tool_output["stdout"]}),
        }

    async def _execute_tool_calls(self, tool_calls: List[dict]):
        return [
            await self._execute_tool_call(tool_call["function"]["name"], tool_call["function"]["arguments"])
            for tool_call in tool_calls
        ]

    async def generate_async(
        self,
        prompt: List,
        tools: List[dict] = None,
        **generation_kwargs,
    ) -> Dict:
        ## FIXME(sanyamk): Will be moved away from here.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "exa_websearch",
                    "description": "Search the web using Exa. Provide relevant links in your answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for Exa.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        assert isinstance(prompt, list), "Only use ChatCompletion API for now."
        assert isinstance(tools, list), "Missing tools specification."

        ## Bookkeeping.
        num_tool_calls = 0
        generation_steps = []
        reasoning_steps = []

        conversation = list(prompt)

        while True:
            generation = await self.model.generate_async(prompt=conversation, tools=tools, **generation_kwargs)

            content = generation.get("generation")
            reasoning_content = generation.get("reasoning_content")
            tool_calls = generation.get("tool_calls", [])

            if content:
                generation_steps.append(content)
                message = {"role": "assistant", "content": content}
                conversation.append(message)

            if reasoning_content:
                reasoning_steps.append(reasoning_content)

            if tool_calls:
                tool_calls_message = await self._parse_tool_calls(tool_calls)
                conversation.append(tool_calls_message)

                tool_calls_output_message = await self._execute_tool_calls(tool_calls_message["tool_calls"])
                conversation.extend(tool_calls_output_message)

                num_tool_calls += len(tool_calls)

                continue

            break

        return {
            "generation": "".join(generation_steps),
            "num_tool_calls": num_tool_calls,
            "reasoning_content": reasoning_steps,
            "conversation": conversation,
        }
