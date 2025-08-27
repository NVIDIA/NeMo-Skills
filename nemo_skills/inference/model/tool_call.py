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
from typing import Dict, List

import yaml
from omegaconf import OmegaConf

from nemo_skills.mcp.config import build_client_manager, resolve_adapters
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

        with open(tool_config_yaml) as f:
            tool_cfg = yaml.safe_load(f)
        tool_cfg.update(additional_config)
        tool_cfg = OmegaConf.create(tool_cfg)

        self.client_manager = build_client_manager(tool_cfg)
        ## FIXME(sanyamk): All these need to be cohesive, so might as well be a single class.
        self.schema_adapter, self.call_interpreter, self.response_formatter = resolve_adapters(tool_cfg)

    async def _execute_tool_call(self, tool_call):
        ## TODO(sanyamk): The correct key format needs to be cohesive with other formatters.
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        try:
            tool_args = json.loads(tool_args)
        except json.decoder.JSONDecodeError as e:
            LOG.exception(e)
            return {"error": "Tool argument parsing failed."}

        try:
            result = await self.client_manager.execute_tool(tool_name, tool_args)
        except Exception as e:
            LOG.exception(e)
            return {"error": "Tool execution failed."}

        return result

    async def _execute_tool_calls(self, tool_calls: List):
        tasks = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
        tool_results = await asyncio.gather(*tasks)
        return [
            self.response_formatter.format(tool_call["function"]["name"], tool_result)
            for tool_call, tool_result in zip(tool_calls, tool_results)
        ]

    async def generate_async(
        self,
        prompt: List,
        tools: List[dict] = None,
        **generation_kwargs,
    ) -> Dict:
        assert isinstance(prompt, list), "Only use ChatCompletion API for now."

        assert tools is None, "Specify ++tool_config=</path/to/file.yaml> only."

        tools = self.schema_adapter.convert(await self.client_manager.list_all_tools())

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
                tool_calls_message = self.call_interpreter.parse(tool_calls)
                conversation.append(tool_calls_message)

                ## TODO(sanyamk): refactor to not rely on hardcoded dict keys.
                tool_calls_output_messages = await self._execute_tool_calls(tool_calls_message["tool_calls"])
                conversation.extend(tool_calls_output_messages)

                num_tool_calls += len(tool_calls)

                continue

            break

        return {
            "generation": "".join(generation_steps),
            "num_tool_calls": num_tool_calls,
            "reasoning_content": reasoning_steps,
            "conversation": conversation,
        }
