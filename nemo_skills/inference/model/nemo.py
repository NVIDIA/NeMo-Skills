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

from .base import BaseModel, BaseRewardModel
from .utils import RequestException, trim_after_stop_phrases


class NemoModel(BaseModel):
    def _generate_single(
        self,
        prompt: str | list,
        tokens_to_generate: int | list[int] = 512,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | list[list[str]] | None = None,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,  # Ignored for Nemo
        extra_body: dict = None,
    ) -> list[dict]:
        """If the engine supports inflight-batching of requests, you only need to define this method.

        We will call it in threads on the list of prompts.
        """
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")
        if top_logprobs is not None:
            raise NotImplementedError("Nemo server does not support top_logprobs parameter.")
        if timeout is not None:
            raise NotImplementedError("Nemo server does not support timeout parameter.")
        if isinstance(prompt, dict):
            raise NotImplementedError("NeMo server does not support OpenAI \"messages\" as prompt.")
        if stream:
            raise NotImplementedError("NeMo server does not support streaming.")
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "sentences": [prompt],
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
        }
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        output = generations['sentences'][0]
        # when the prompt starts from special tokens like bos, nemo will remove them,
        # so we need this hack to find where to start the cut
        begin_idx = 0
        while begin_idx < len(prompt) and not prompt[begin_idx:].startswith(output[:20]):
            begin_idx += 1
        output = {'generation': output[(len(prompt) - begin_idx) :]}
        return output

    def generate(
        self,
        prompts: list[str | list],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,  # Ignored for Nemo
        extra_body: dict = None,
    ) -> list[dict]:
        # we are overriding generate directly, since nemo doesn't support inflight batching
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")
        if top_logprobs is not None:
            raise NotImplementedError("Nemo server does not support top_logprobs parameter.")
        if timeout is not None:
            raise NotImplementedError("Nemo server does not support timeout parameter.")

        if isinstance(prompts[0], dict):
            raise NotImplementedError("NeMo server does not support OpenAI \"messages\" as prompt.")
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "sentences": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
            "stream": stream,
        }
        self.preprocess_request(request)
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        outputs = [None] * len(generations['sentences'])
        for idx, generation in enumerate(generations['sentences']):
            # when the prompt starts from special tokens like bos, nemo will remove them,
            # so we need this hack to find where to start the cut
            begin_idx = 0
            while begin_idx < len(prompts[idx]) and not prompts[idx][begin_idx:].startswith(generation[:20]):
                begin_idx += 1
            outputs[idx] = {'generation': generation[(len(prompts[idx]) - begin_idx) :]}

        if remove_stop_phrases:
            for output in outputs:
                output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

        # TODO: return num_generated_tokens as well
        return outputs


class NemoRewardModel(BaseRewardModel):
    def score(self, prompts: list[str]) -> list[float]:
        request = {
            "prompts": prompts,
        }
        response = self.requests_lib.post(f"http://{self.server_host}:{self.server_port}/score", json=request)

        if response.status_code != 200:
            raise RequestException(f"Failed to score prompts: {response.text}")

        scores = response.json()

        outputs = [{"generation": score} for score in scores["rewards"]]
        return outputs
