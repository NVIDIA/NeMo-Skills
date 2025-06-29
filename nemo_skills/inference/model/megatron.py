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

from openai import Stream

from .base import OpenAIAPIModel
from .utils import trim_after_stop_phrases


class MegatronModel(OpenAIAPIModel):
    def __init__(self, **kwargs):
        # Megatron uses a non-standard base URL (no /v1) and a fixed model name.
        super().__init__(model="model", use_v1_endpoint=False, **kwargs)

    def _build_chat_request_params(self, **kwargs) -> dict:
        raise NotImplementedError("Megatron server does not support chat completions.")

    def _build_completion_request_params(
        self,
        prompt: str,
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
        **kwargs,
    ) -> dict:
        # Parameter validation specific to Megatron
        if stream:
            raise NotImplementedError("Megatron server does not support streaming.")
        if min_p > 0:
            raise NotImplementedError("Megatron server does not support min_p parameter.")
        if repetition_penalty != 1.0:
            raise NotImplementedError("Megatron server does not support repetition_penalty parameter.")
        if top_k != 0:
            raise NotImplementedError("Megatron server does not support top_k parameter.")

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
        }

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
        **kwargs,
    ) -> dict | Stream:
        # Overriding generate to provide its own batching support, bypassing the parent's async logic.
        params = self._build_completion_request_params(
            prompt=None,  # Not used for batch call
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            top_logprobs=top_logprobs or 0,
            timeout=timeout,
            stream=stream,
        )
        params["prompt"] = prompts  # Replace single prompt with batch

        response = self.client.completions.create(**params)
        outputs = self.parse_openai_response(response, batch=True, top_logprobs=top_logprobs)

        if remove_stop_phrases:
            for output in outputs:
                output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases or [])

        return outputs

    @classmethod
    def parse_openai_response(
        cls, response: "openai.types.Completion", batch: bool = False, top_logprobs: int | None = None
    ) -> dict | list[dict]:
        """Parse OpenAI response to extract the generated text and other metadata.

        Args:
            response: The response from OpenAI API
            batch: Whether the response contains multiple generations (batch mode)

        Returns:
            A single dict with generation info or a list of dicts for batch mode
        """

        def process_choice(choice, top_logprobs: int | None = None):
            output = choice.text
            # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
            if choice.finish_reason == "stop":
                if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                    output += choice.stop_reason
                # sglang has a little different api here
                if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                    output += choice.matched_stop

            result = {'generation': output, 'num_generated_tokens': -1}
            if choice.logprobs and choice.logprobs.tokens:  # logprobs is always populated, but empty if not requested
                if top_logprobs is not None and top_logprobs != 0:
                    result['logprobs'] = choice.logprobs.token_logprobs
                    result['tokens'] = choice.logprobs.tokens
                    result['top_logprobs'] = choice.logprobs.top_logprobs
                result['num_generated_tokens'] = len(choice.logprobs.tokens)
            return result

        if batch:
            return [process_choice(choice, top_logprobs) for choice in response.choices]
        else:
            assert not isinstance(response, list)
            assert len(response.choices) == 1
            return process_choice(response.choices[0], top_logprobs)
