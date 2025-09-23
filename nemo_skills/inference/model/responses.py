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

import logging

from openai import AsyncOpenAI, OpenAI

from nemo_skills.utils import get_logger_name

from .base import BaseModel
from .context_retry import with_context_retry

LOG = logging.getLogger(get_logger_name(__file__))


class ResponsesModel(BaseModel):
    """Model implementation using OpenAI Responses API via LiteLLM.

    This model uses the responses API endpoint instead of chat completions,
    which is useful for models like gpt-oss that support the responses format.
    """

    MODEL_PROVIDER = "openai"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize OpenAI clients for responses API
        self.client = OpenAI(base_url=self.base_url, api_key=self.litellm_kwargs["api_key"])
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.litellm_kwargs["api_key"])

    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """Build request parameters for responses API with string input."""
        request = {
            "input": prompt,
            "max_output_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "extra_body": {
                "seed": random_seed,
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
                "stop": stop_phrases,
                "top_logprobs": top_logprobs,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                **(extra_body or {}),
            },
        }

        # Only include tools if they are provided
        if tools is not None:
            request["tools"] = tools

        return request

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict = None,
    ) -> dict:
        """Build request parameters for responses API with messages input."""
        request = {
            "input": messages,
            "max_output_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "extra_body": {
                "seed": random_seed,
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
                "stop": stop_phrases,
                "top_logprobs": top_logprobs,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                **(extra_body or {}),
            },
        }

        # Only include tools if they are provided
        if tools is not None:
            request["tools"] = tools

        return request

    def _parse_responses_response(self, response, include_response: bool = False, **kwargs) -> dict:
        """Parse responses API response into standard format."""

        result = {"generation": "", "num_generated_tokens": 0}

        # Get token usage
        if hasattr(response, "usage"):
            result["num_generated_tokens"] = getattr(response.usage, "output_tokens", 0)

        # Check for tool calls in the output array
        tool_calls = []
        reasoning_content = ""
        generation_text = ""

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                # Handle reasoning content
                if hasattr(output_item, "type") and output_item.type == "reasoning":
                    if hasattr(output_item, "content") and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, "text"):
                                reasoning_content += content_item.text + "\n"

                # Handle function calls
                elif hasattr(output_item, "type") and output_item.type == "function_call":
                    tool_calls.append(output_item)

                # Handle message content
                elif hasattr(output_item, "type") and output_item.type == "message":
                    if hasattr(output_item, "content") and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, "text"):
                                generation_text += content_item.text

        # Set the appropriate response fields
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["generation"] = ""  # No text generation when there are tool calls
        else:
            result["generation"] = generation_text

        # Add reasoning content if available
        if reasoning_content:
            result["reasoning_content"] = reasoning_content.strip()

        # Add finish reason if available
        if hasattr(response, "status"):
            result["finish_reason"] = response.status

        # Add serialized output for conversation history
        result["serialized_output"] = self._serialize_response_output(response)

        if include_response:
            result["response"] = response

        return result

    def _serialize_response_output(self, response) -> list[dict]:
        """Serialize response output objects using model_dump() for conversation history."""
        serialized_output = []

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                try:
                    # Try to use model_dump() method if available (Pydantic models)
                    if hasattr(output_item, "model_dump"):
                        serialized_output.append(output_item.model_dump())
                    # Fallback to dict conversion
                    elif hasattr(output_item, "__dict__"):
                        serialized_output.append(output_item.__dict__)
                    # Last resort: convert to string representation
                    else:
                        serialized_output.append({"content": str(output_item), "type": "unknown"})
                except Exception as e:
                    LOG.warning(f"Failed to serialize output item: {e}")
                    # Fallback serialization
                    serialized_output.append({"content": str(output_item), "type": "error", "error": str(e)})

        return serialized_output

    @with_context_retry
    async def generate_async(
        self,
        prompt: str | list[dict],
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """Generate response using responses API."""

        # Check tool calls are a list of dict
        if tools is not None:
            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool must be a dictionary, got {type(tool)}")

        kwargs = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "random_seed": random_seed,
            "stop_phrases": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "extra_body": extra_body,
        }

        request_params = self._build_request_params(prompt=prompt, stream=stream, **kwargs)

        # Use OpenAI client for responses API
        LOG.info(f"Making responses API call with params: {request_params}")
        LOG.info(f"Full litellm_kwargs: {self.litellm_kwargs}")
        LOG.info(f"Model name from litellm_kwargs: {self.litellm_kwargs['model']}")
        LOG.info(f"Base URL: {self.base_url}")

        # Use the original model name (without litellm prefix) for OpenAI client
        model_name = self.model_name_or_path  # Just gpt-oss-20b
        LOG.info(f"About to call responses.create with model: {model_name}")

        response = await self.async_client.responses.create(model=model_name, **request_params)

        LOG.info(f"Response from server: {response}")
        LOG.info(f"Response type: {type(response)}")

        if stream:
            # Handle streaming responses
            return self._stream_responses_chunks_async(response)
        else:
            result = self._parse_responses_response(response, include_response=include_response, **kwargs)
            self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
            return result

    @with_context_retry
    def generate_sync(
        self,
        prompt: str | list[dict],
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """Synchronous version of generate using responses API."""

        # Check tool calls are a list of dict
        if tools is not None:
            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool must be a dictionary, got {type(tool)}")

        kwargs = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "random_seed": random_seed,
            "stop_phrases": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "extra_body": extra_body,
        }

        request_params = self._build_request_params(prompt=prompt, stream=stream, **kwargs)

        # Use OpenAI client for responses API
        LOG.info(f"Making sync responses API call with params: {request_params}")
        LOG.info(f"Model name: {self.litellm_kwargs['model']}")

        # Use the original model name (without litellm prefix) for OpenAI client
        model_name = self.model_name_or_path  # Just gpt-oss-20b

        response = self.client.responses.create(model=model_name, **request_params)

        LOG.info(f"Response from server: {response}")
        LOG.info(f"Response type: {type(response)}")

        if stream:
            # Handle streaming responses
            return self._stream_responses_chunks_sync(response)
        else:
            result = self._parse_responses_response(response, include_response=include_response, **kwargs)
            self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
            return result

    def _stream_responses_chunks_sync(self, response):
        """Synchronous version of stream responses chunks."""
        for chunk in response:
            result = self._process_responses_chunk(chunk)
            if result:
                yield result

    async def _stream_responses_chunks_async(self, response):
        """Async version of stream responses chunks."""
        async for chunk in response:
            result = self._process_responses_chunk(chunk)
            if result:
                yield result

    def _process_responses_chunk(self, chunk):
        """Process a single responses API chunk."""
        # This will depend on the actual streaming format from responses API
        # For now, implement basic text streaming
        if hasattr(chunk, "output_text"):
            return {"generation": chunk.output_text or ""}
        elif hasattr(chunk, "output") and chunk.output:
            if isinstance(chunk.output, list) and len(chunk.output) > 0:
                first_output = chunk.output[0]
                if hasattr(first_output, "text"):
                    return {"generation": first_output.text or ""}

        # Fallback
        return {"generation": ""}
