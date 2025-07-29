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
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
from openai import BadRequestError

from nemo_skills.utils import get_logger_name

from .base import BaseRewardModel, OpenAIAPIModel

LOG = logging.getLogger(get_logger_name(__file__))


class VLLMModel(OpenAIAPIModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_request_body(self, top_k, min_p, repetition_penalty, extra_body: dict = None):
        full_extra_body = {
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "spaces_between_special_tokens": False,
        }

        if top_k > 0:
            full_extra_body["top_k"] = top_k

        if extra_body:
            full_extra_body.update(extra_body)

        return full_extra_body

    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict = None,
    ) -> dict:
        return {
            "model": self.model,
            "prompt": [prompt],
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs,
            "stream": stream,
            "echo": False,
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
        }

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
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
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs is not None,
            "top_logprobs": top_logprobs,
            "n": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": stream,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
        }
        if tools is not None:
            request["tools"] = tools
        return request


class VLLMRewardModel(BaseRewardModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ssh_server and self.ssh_key_path:
            raise NotImplementedError("SSH tunnelling is not implemented for vLLM model.")

        self.oai_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.server_host}:{self.server_port}/v1",
            timeout=None,
        )

        # Reward models are accessed via the "pooling" interface
        # https://docs.vllm.ai/en/latest/models/pooling_models.html
        self.request_url = f"http://{self.server_host}:{self.server_port}/pooling"

        model_list = self.oai_client.models.list()
        self.model = model_list.data[0].id

    def _score_single_prompt(self, prompt):
        """Score a single prompt"""

        per_token_scores = None
        inference_error = ""
        try:
            response = requests.post(self.request_url, json={"input": prompt, "model": self.model})
            output = response.json()
            per_token_scores = output['data'][0]['data']
        except requests.exceptions.HTTPError as err:
            inference_error = f"Request failed: {err}"
        except ValueError as ve:
            # Could be that the sequence exceeds the maximum context length
            inference_error = f"Tokenization error: {ve}"
        except KeyError as ke:
            # Returned output is not adhering to the expected output format
            inference_error = f"Output fmt error: {ke}\n{output}"

        if inference_error:
            LOG.warning(inference_error)

        if per_token_scores is None:
            # Return a trivial reward model score
            return {"generation": 0.0, "inference_error": inference_error}

        last_token_score = per_token_scores[-1]
        score = None
        if self.model_type == "orm":
            # Last token's score
            if isinstance(last_token_score, list):
                logit_score = last_token_score[0]
            else:
                logit_score = last_token_score
            # Normalize the score
            score = 1 / (1 + math.exp(-logit_score))
        elif self.model_type == "prm":
            # Last token's score, a 2-entry array where the second entry is the probability of being correct
            score = last_token_score[1]

        return {"generation": score}

    def score(self, prompts: list[str]) -> list[float]:
        outputs = [None] * len(prompts)  # Pre-allocate a list to store results in correct order
        futures = {}

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            for idx, prompt in enumerate(prompts):
                futures[executor.submit(self._score_single_prompt, prompt)] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except BadRequestError as e:
                    error_details = e.body
                    error_message = error_details.get("message", "No message found")
                    error_code = error_details.get("code", "No code found")
                    if error_code == 400 and 'maximum context length' in error_message:
                        outputs[idx] = {
                            "generation": 0
                        }  # Default value set as 0 if we have request over maximum context length
                        LOG.warning("Maximum context length exceeded, setting reward score as 0")
                    else:
                        raise
        return outputs
