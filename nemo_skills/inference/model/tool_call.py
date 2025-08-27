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
import json
import logging
import textwrap
from typing import Dict, List

import yaml
from litellm.types.utils import ChatCompletionMessageToolCall
from omegaconf import OmegaConf

from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.mcp.config import MCPConfig, build_client_manager, resolve_adapters
from nemo_skills.utils import get_logger_name

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


class ToolCallingWrapper:
    """
    Wrapper to handle tool calling.

    TODO(sanyamk): Supports only Chat Completions API for now.
    """

    def __init__(self, model: BaseModel, tool_config_yaml: str, additional_config: dict):
        self.model = model
        tool_cfg = yaml.safe_load(tool_config_yaml)
        tool_cfg.update(additional_config)
        tool_cfg = OmegaConf.create(tool_cfg)
        self.client_manager = build_client_manager(tool_cfg)
        self.schema_adapter, self.call_interpreter, self.response_formatter = resolve_adapters(tool_cfg)

    async def _parse_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCall]):
        """Convert tool calls to conversation message item."""

        tool_calls = [self.call_interpreter.parse(tool_call) for tool_call in tool_calls]

        return {"role": "assistant", "tool_calls": tool_calls}

    ## TODO(sanyamk): Support tool call IDs.
    async def _execute_tool_call(self, tool_args):
        return await self.manager.execute_tool(*tool_args)

    async def _execute_tool_calls(self, tool_calls: List[dict]):
        tasks = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
        return await asyncio.gather(*tasks)

    async def generate_async(
        self,
        prompt: List,
        tools: List[dict] = None,
        **generation_kwargs,
    ) -> Dict:
        ## FIXME(sanyamk): Will be moved away from here.
        tools = await self.client_manager.list_all_tools()
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
