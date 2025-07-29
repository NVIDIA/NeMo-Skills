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

from .base import BaseModel
from .utils import trim_after_stop_phrases


class TRTLLMModel(BaseModel):
    def _generate_single_base(
        self,
        prompt: str | list,
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
        generate_endpoint: str = "generate",
        stream: bool = False,
        # Ignored for TRTLLM. TODO: disallow setting
        reasoning_effort: str | list[int] | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> list[dict]:
        if isinstance(prompt, dict):
            raise NotImplementedError("trtllm server does not support OpenAI \"messages\" as prompt.")
        if top_logprobs is not None and top_logprobs > 1:
            raise NotImplementedError("This code does not support `top_logprobs` > 1.")
        if stream:
            raise NotImplementedError("trtllm server does not support streaming.")
        if generate_endpoint not in ["generate", "generate_async"]:
            raise ValueError(f"Invalid generate endpoint: {generate_endpoint}")

        if stop_phrases is None:
            stop_phrases = []

        request = {
            "prompt": prompt,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "top_p_min": min_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
        }
        output_dict = self.requests_lib.put(
            url="http://{}:{}/{}".format(self.server_host, self.server_port, generate_endpoint),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()

        if generate_endpoint == "generate":
            return output_dict
        else:
            return output_dict['generation_id']

    def _generate_single_async(self, prompt: str | list, **kwargs):
        """Asynchronous generation."""
        return self._generate_single_base(prompt, generate_endpoint="generate_async", **kwargs)

    def _generate_single(self, prompt: str | list, **kwargs):
        """Synchronous generation."""
        return self._generate_single_base(prompt, generate_endpoint="generate", **kwargs)

    def generate_async(
        self,
        prompts: list[str | list],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        # Ignored for TRTLLM. TODO: disallow setting
        reasoning_effort: str | list[int] | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        kwargs = {
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
            'top_logprobs': top_logprobs,
            'timeout': timeout,
            'stream': stream,
        }
        for key, value in kwargs.items():
            is_list = False
            if key == 'stop_phrases' and (value and isinstance(value[0], list)):
                is_list = True
            if key != 'stop_phrases' and isinstance(value, list):
                is_list = True
            if is_list and len(value) != len(prompts):
                raise ValueError(f"Length of {key} should match the number of prompts.")
            if not is_list:
                kwargs[key] = [value for _ in range(len(prompts))]

        # futures are not really necessary here unless the number of prompts is huge
        # as we get http reply right away with generation id from the trtllm server
        futures = []
        for request_idx in range(len(prompts)):
            request = {key: value[request_idx] for key, value in kwargs.items()}
            request['prompt'] = prompts[request_idx]
            self.preprocess_request(request)
            futures.append(self.executor.submit(self._generate_single_async, **request))
        outputs = [future.result() for future in futures]

        new_gen_id_to_params = {
            gen_id: (req_stop_phrases, remove_stop_phrases)
            for gen_id, req_stop_phrases in zip(outputs, kwargs["stop_phrases"])
        }

        self.gen_id_to_params.update(new_gen_id_to_params)

        return outputs

    def cancel_generations(self, generation_ids: list[str]) -> list[str]:
        statuses = []
        for generation_id in generation_ids:
            request = {
                "generation_id": generation_id,
            }
            output_dict = self.requests_lib.put(
                url="http://{}:{}/cancel_generation".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
            statuses.append(output_dict["status"])

        return statuses

    def get_generations(self, generation_ids: list[str]) -> list[dict]:
        generations = []
        for generation_id in generation_ids:
            request = {
                "generation_id": generation_id,
            }
            output = self.requests_lib.put(
                url="http://{}:{}/get_generation".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
            stop_phrases, remove_stop_phrases = self.gen_id_to_params[generation_id]
            if remove_stop_phrases:
                if output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations
