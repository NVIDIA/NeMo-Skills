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
from typing import Any, Dict, List

from litellm.types.utils import ChatCompletionMessageToolCall


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


class ConversationManager(ABC):
    """Manages conversation history building for different model types."""

    @abstractmethod
    def add_user_message(self, conversation: List[Dict[str, Any]], content: str) -> None:
        """Add a user message to the conversation."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def add_assistant_response(self, conversation: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        """Add an assistant response to the conversation."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def add_tool_results(
        self, conversation: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> None:
        """Add tool call results to the conversation."""
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
                    "name": t["name"],
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


class CompletionResponseFormatter(ToolResponseFormatter):
    # https://qwen.readthedocs.io/en/latest/framework/function_call.html#id2
    def format(self, tool_call: ChatCompletionMessageToolCall, result):
        return {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }


class ChatCompletionCallInterpreter(ToolCallInterpreter):
    """Convert tool calls to a chat message item.

    Should be broadly compatible with any OpenAI-like APIs,
    and HuggingFace Chat templates.

    NOTE(sanyamk): For error handling, delay JSON parsing of arguments to the model class.
    """

    def parse(self, tool_calls: List[ChatCompletionMessageToolCall]):
        tool_calls = [
            {
                "type": tool_call.type,
                "id": tool_call.id,
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
            for tool_call in tool_calls
        ]

        return {"role": "assistant", "tool_calls": tool_calls}


class ChatCompletionResponseFormatter(ToolResponseFormatter):
    """Convert tool call result to chat message item.

    Use in conjunction with `ChatCompletionCallInterpreter`.
    """

    def format(self, tool_call, result):
        return {
            "role": "tool",
            "name": tool_call["function"]["name"],
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result) if not isinstance(result, str) else result,
        }


# ==============================
# CONVERSATION MANAGERS
# ==============================


class CompletionConversationManager(ConversationManager):
    """Manages conversation history for chat completion models."""

    def add_user_message(self, conversation: List[Dict[str, Any]], content: str) -> None:
        """Add a user message to the conversation."""
        conversation.append({"role": "user", "content": content})

    def add_assistant_response(self, conversation: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        """Add an assistant response to the conversation."""
        message = {"role": "assistant", "content": response["generation"]}

        # Add reasoning content if available
        if "reasoning_content" in response:
            message["reasoning_content"] = response["reasoning_content"]

        conversation.append(message)

    def add_tool_results(
        self, conversation: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> None:
        """Add tool call results to the conversation."""
        # Update the last assistant message with tool calls
        if conversation and conversation[-1]["role"] == "assistant":
            conversation[-1]["tool_calls"] = tool_calls

        # Add tool result messages
        for tool_call, result in zip(tool_calls, results):
            conversation.append(
                {
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                }
            )


class ResponsesConversationManager(ConversationManager):
    """Manages conversation history for responses API models."""

    def add_user_message(self, conversation: List[Dict[str, Any]], content: str) -> None:
        """Add a user message to the conversation."""
        conversation.append({"role": "user", "content": content})

    def add_assistant_response(self, conversation: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        """Add an assistant response to the conversation using serialized output."""
        # Use the serialized output from the responses API
        if "serialized_output" in response:
            conversation.extend(response["serialized_output"])
        else:
            # Fallback to basic message format
            conversation.append({"role": "assistant", "content": response["generation"]})

    def add_tool_results(
        self, conversation: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> None:
        """Add tool call results to the conversation."""
        # For responses API, add tool results as function_call_output items
        for tool_call, result in zip(tool_calls, results):
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.get("id", "unknown"),
                    "output": json.dumps(result) if not isinstance(result, str) else result,
                }
            )


# ==============================
# RESPONSES API ADAPTERS
# ==============================


class ResponsesCallInterpreter(ToolCallInterpreter):
    """Convert responses API tool calls to a standardized format."""

    def parse(self, tool_calls: List[Any]) -> Dict[str, Any]:
        """Parse tool calls from responses API format."""
        parsed_calls = []

        for tool_call in tool_calls:
            parsed_call = {
                "type": "function",
                "id": getattr(tool_call, "call_id", getattr(tool_call, "id", "unknown")),
                "function": {
                    "name": getattr(tool_call, "name", "unknown"),
                    "arguments": getattr(tool_call, "arguments", "{}"),
                },
            }
            parsed_calls.append(parsed_call)

        return {"role": "assistant", "tool_calls": parsed_calls}


class ResponsesResponseFormatter(ToolResponseFormatter):
    """Format tool call results for responses API conversation history."""

    def format(self, tool_call: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response from a tool call for responses API."""
        return {
            "type": "function_call_output",
            "call_id": tool_call["id"],
            "output": json.dumps(result) if not isinstance(result, str) else result,
        }


class ResponsesSchemaAdapter(ToolSchemaAdapter):
    """Convert MCP tool definitions to responses API format (flatter structure)."""

    def convert(self, tools):
        """Convert tools to responses API format without nested 'function' object."""
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]
