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
import os
import re

from .base import OpenAIAPIModel


class OpenAIModel(OpenAIAPIModel):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        **kwargs,
    ):
        model = model or os.getenv("NEMO_SKILLS_OPENAI_MODEL")
        if model is None:
            raise ValueError("model argument is required for OpenAI model.")

        if base_url is None:
            base_url = os.getenv("NEMO_SKILLS_OPENAI_BASE_URL", f"http://{host}:{port}/v1")

        if api_key is None:
            if 'api.nvidia.com' in base_url:
                api_key = os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY is required for NVIDIA models and could not be found.")
            elif 'api.openai.com' in base_url:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI models and could not be found.")

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            **kwargs,
        )

    def preprocess_request(self, request: dict):
        """OpenAI doesn't use top_k, so we don't apply the greedy conversion."""
        pass

    def _is_reasoning_model(self, model_name: str) -> bool:
        return re.match(r"^o\d", model_name)

    def _build_completion_request_params(self, **kwargs) -> dict:
        raise NotImplementedError("OpenAIModel only supports chat completions. Use a list of messages as a prompt.")

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
    ) -> dict:
        # Validations
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to 0.")
        if min_p > 0:
            raise ValueError("`min_p` is not supported by OpenAI API, please set it to 0.0.")
        if stream and top_logprobs is not None:
            raise ValueError("`top_logprobs` is not supported with stream=True.")

        params = {
            "model": self.model,
            "messages": messages,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "timeout": timeout,
            "stream": stream,
        }

        if self._is_reasoning_model(self.model):
            # Reasoning model specific validations and parameters
            if temperature != 0.0:
                raise ValueError(
                    "`temperature` is not supported by reasoning models, please set it to default value `0.0`."
                )
            if top_p != 0.95:
                raise ValueError(
                    "`top_p` is not supported by reasoning models, please set it to default value `0.95`."
                )
            if repetition_penalty != 1.0:
                raise ValueError(
                    "`repetition_penalty` is not supported by reasoning models, please set it to default value `1.0`."
                )
            if top_logprobs is not None:
                raise ValueError("`top_logprobs` is not supported by reasoning models, please set it to `None`.")

            params["max_completion_tokens"] = tokens_to_generate
            params["messages"] = [
                {**msg, "role": "developer"} if msg.get("role") == "system" else msg for msg in messages
            ]
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
        else:
            # Standard model parameters
            if reasoning_effort is not None:
                raise ValueError("`reasoning_effort` is only supported by reasoning models.")
            params["presence_penalty"] = repetition_penalty
            params["logprobs"] = top_logprobs is not None
            params["top_logprobs"] = top_logprobs
            params["max_tokens"] = tokens_to_generate
            params["temperature"] = temperature
            params["top_p"] = top_p

        return params

    def batch_generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        stop_phrases: list[str] | None = None,
        reasoning_effort: str | list[int] | None = None,
    ) -> list[dict]:
        # only supported by the OpenAI endpoint!
        if stop_phrases is None:
            stop_phrases = []

        # Prepare batch requests using _build_request_params
        with open("requests.jsonl", "wt", encoding='utf-8') as fout:
            for idx, prompt in enumerate(prompts):
                # Reuse the existing parameter building logic
                params = self._build_chat_request_params(
                    messages=prompt,
                    tokens_to_generate=tokens_to_generate,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=0.0,  # Default value
                    repetition_penalty=repetition_penalty,
                    random_seed=random_seed,
                    stop_phrases=stop_phrases,
                    top_logprobs=top_logprobs,
                    reasoning_effort=reasoning_effort,
                    stream=False,  # not supported in batch
                    timeout=None,  # not supported in batch
                )

                fout.write(
                    json.dumps(
                        {
                            "custom_id": f"{idx}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": params,
                        }
                    )
                    + "\n"
                )

        with open("requests.jsonl", "rb") as batch_file_handle:
            batch_file_id = self.client.files.create(file=batch_file_handle, purpose="batch").id
            metadata = self.client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",  # the only supported value, but should finish faster
                metadata={"description": "batch job"},
            )

        return metadata

    def get_batch_results(self, batch_id):
        metadata = self.client.batches.retrieve(batch_id)
        outputs = None
        if metadata.status == 'completed' and metadata.output_file_id is not None:
            file_response = self.client.files.content(metadata.output_file_id)
            responses = file_response.text
            outputs = []
            for line in responses.split('\n')[:-1]:
                data = json.loads(line)
                outputs.append(
                    {
                        'custom_id': data['custom_id'],
                        'generation': data['response']['body']['choices'][0]['message']['content'],
                    }
                )
            outputs = sorted(outputs, key=lambda x: int(x['custom_id']))
            for output in outputs:
                output.pop('custom_id')

        return metadata, outputs
