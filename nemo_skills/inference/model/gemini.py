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

import os
import httpx
import litellm
from .openai import OpenAIModel


class GeminiModel(OpenAIModel):
    MODEL_PROVIDER = "gemini"

    def __init__(
        self,
        *args,
        model: str,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        model:
            - gemini-2.5-pro: thinking budget 128-32768 (default: dynamic thinking)
            - gemini-2.5-flash: thinking budget 0-24576 (default: dynamic thinking)
            - gemini-2.5-flash-lite: thinking budget 0-24576 (default: no thinking)
        """
        assert os.getenv("GEMINI_API_KEY") is not None, "GEMINI_API_KEY is not set"
        model_litellm = f"{self.MODEL_PROVIDER}/{model}"
        self.model = model
        self.litellm_kwargs = dict(
            model=model_litellm,
            max_retries=max_retries,
        )
        httpx_limits = httpx.Limits(
            max_keepalive_connections=2048, max_connections=2048
        )
        litellm.client_session = httpx.Client(limits=httpx_limits)
        litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

    def _build_chat_request_params(
        self,
        messages: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
        timeout: int | None,
        top_logprobs: int | None,
        stream: bool,
        reasoning_effort: str | None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """
        https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/constants.py#L45-L56
        reasoning_effort:
            - None: thinking budget tokens: 0
            - low: maximum thinking budget tokens: 1024 (env var: DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET)
            - medium: maximum thinking budget tokens: 2048 (env var: DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET)
            - high: maximum thinking budget tokens: 4096 (env var: DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET)
            - dynamic: maximum thinking budget tokens: -1
        """

        # Vertext AI params: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
        # litellm default params: https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/gemini/chat/transformation.py#L73-L90
        # litellm other params: https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L147-L174
        params = {
            "messages": messages,
            "stop": stop_phrases or None,
            "timeout": timeout,
            "stream": stream,
            "tools": tools,
            "max_completion_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": top_logprobs is not None,
            "top_k": top_k,
            "seed": random_seed,
            "presence_penalty": repetition_penalty,
            "top_logprobs": top_logprobs,
            "allowed_openai_params": ['top_k', 'seed', 'presence_penalty', 'top_logprobs'],
        }

        if reasoning_effort is None:
            # https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L438-L442
            reasoning_effort = "disable"

        elif reasoning_effort == "dynamic":
            reasoning_effort = None
            # https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L451-L465
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": -1,
            }
        
        if self.model == "gemini-2.5-pro":
            assert reasoning_effort != "disable", "Gemini 2.5 Pro cannnot disable reasoning, please set reasoning_effort to ['dynamic', 'low', 'medium', 'high']"
        
        params["reasoning_effort"] = reasoning_effort
        
        return params