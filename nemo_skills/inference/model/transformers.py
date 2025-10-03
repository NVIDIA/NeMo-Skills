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

from nemo_skills.utils import get_logger_name

from .context_retry import with_context_retry
from .vllm import VLLMModel

LOG = logging.getLogger(get_logger_name(__file__))


class TransformersModel(VLLMModel):
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
        if top_k > 0:
            raise ValueError("top_k is not supported for transformers server")
        if min_p != 0.0:
            raise ValueError("min_p is not supported for transformers server")
        if repetition_penalty != 1.0:
            raise ValueError("repetition_penalty is not supported for transformers server")
        if extra_body:
            raise ValueError("extra_body is not supported for transformers server")

        request = {
            "messages": messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs is not None,
            "top_logprobs": top_logprobs,
            "n": 1,
            # somehow transformers return some nonsense in non-streaming mode (like a big string of streaming chunks)
            # so we just always use streaming and wait for the end
            "stream": True,
            "timeout": timeout,
            "tools": tools,
        }
        if reasoning_effort:
            request["allowed_openai_params"] = ["reasoning_effort"]
            request["reasoning_effort"] = reasoning_effort
        return request

    # no sync version as it's deprecated anyway

    async def _accumulate_chat_chunks_async(self, response):
        """Accumulate all streaming chunks into a final response object."""
        full_generation = ""
        full_reasoning = ""
        finish_reason = None

        async for chunk in response:
            if chunk.get("generation"):
                full_generation += chunk["generation"]
            if chunk.get("reasoning_content"):
                full_reasoning += chunk["reasoning_content"]
            if chunk.get("finish_reason"):
                finish_reason = chunk["finish_reason"]

        final_result = {"generation": full_generation}

        if full_reasoning:
            final_result["reasoning_content"] = full_reasoning

        if finish_reason:
            final_result["finish_reason"] = finish_reason

        return final_result

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
        """
        Async version of generate for single prompt.
        Transformers backend always uses streaming internally for chat requests.
        """
        if isinstance(prompt, str):
            raise NotImplementedError("Transformers server does not support text completions.")

        # For chat requests, we need special handling since transformers server
        # doesn't work properly in non-streaming mode
        user_wants_stream = stream

        # Call parent with stream=True to force streaming internally
        result = await super().generate_async(
            prompt=prompt,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            top_logprobs=top_logprobs,
            timeout=timeout,
            remove_stop_phrases=False,  # We'll handle this ourselves after accumulation
            stream=True,  # Always stream internally
            reasoning_effort=reasoning_effort,
            tools=tools,
            include_response=include_response,
            extra_body=extra_body,
        )

        # If user wants streaming, just return the generator
        if user_wants_stream:
            return result

        # Otherwise, accumulate all chunks into a single result
        accumulated = await self._accumulate_chat_chunks_async(result)
        self._maybe_apply_stop_phrase_removal(accumulated, remove_stop_phrases, stop_phrases)
        return accumulated
