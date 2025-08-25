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
from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionMessageToolCall


# ==============================
# ADAPTER INTERFACES
# ==============================
class ToolSchemaAdapter(ABC):
    @abstractmethod
    def convert(self, tools: list[dict]) -> list[dict]:
        """Convert MCP tool definitions into model-specific schema."""
        raise NotImplementedError("Subclasses must implement this method.")


class ToolCallInterpreter(ABC):
    @abstractmethod
    def parse(self, raw_call: dict) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")


class ToolResponseFormatter(ABC):
    @abstractmethod
    def format(self, tool_call: ChatCompletionMessageToolCall, result: dict) -> dict:
        """Format the response from a tool call."""
        raise NotImplementedError("Subclasses must implement this method.")


# ==============================
# ADAPTER IMPLEMENTATIONS
# ==============================


class OpenAISchemaAdapter(ToolSchemaAdapter):
    # https://platform.openai.com/docs/guides/function-calling#defining-functions
    def convert(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{t['name']}",
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]


class OpenAICallInterpreter(ToolCallInterpreter):
    def parse(self, tool_call):
        fn = tool_call.function
        tool = fn.name
        return {"tool_name": tool, "args": json.loads(fn.arguments)}


class OpenAIResponseFormatter(ToolResponseFormatter):
    # https://platform.openai.com/docs/guides/function-calling
    def format(self, tool_call: ChatCompletionMessageToolCall, result):
        return {
            "type": "function_call_output",
            "call_id": tool_call.id,
            "output": json.dumps(result),
        }


class QwenResponseFormatter(ToolResponseFormatter):
    # https://qwen.readthedocs.io/en/latest/framework/function_call.html#id2
    def format(self, tool_call: ChatCompletionMessageToolCall, result):
        return {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }


# ==============================
# REGISTRY
# ==============================
class AdapterRegistry:
    def __init__(self):
        self.schemas = {}
        self.interpreters = {}
        self.response_formatters = {}

    def register_schema(self, model_type: str, adapter: ToolSchemaAdapter):
        self.schemas[model_type] = adapter

    def register_interpreter(self, model_type: str, adapter: ToolCallInterpreter):
        self.interpreters[model_type] = adapter

    def register_response_formatter(self, model_type: str, adapter: ToolResponseFormatter):
        self.response_formatters[model_type] = adapter

    def get_schema(self, model_type: str):
        return self.schemas[model_type]

    def get_interpreter(self, model_type: str):
        return self.interpreters[model_type]

    def get_response_formatter(self, model_type: str):
        return self.response_formatters[model_type]


# ---- Example setup
registry = AdapterRegistry()

registry.register_schema("openai", OpenAISchemaAdapter())
registry.register_interpreter("openai", OpenAICallInterpreter())
registry.register_response_formatter("openai", OpenAIResponseFormatter())

# Qwen can reuse OpenAI
registry.register_schema("qwen", OpenAISchemaAdapter())
registry.register_interpreter("qwen", OpenAICallInterpreter())
registry.register_response_formatter("qwen", QwenResponseFormatter())
