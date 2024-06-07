# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import abc
import json
import logging
import os
import re
import time
from typing import Union

import requests

LOG = logging.getLogger(__name__)


def remove_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    return re.split("|".join([sp.replace('|', '\\|') for sp in stop_phrases]), text, maxsplit=1)[0]


def preprocess_request(request: dict):
    """Just a small utility to pre-process some of the parameters of request."""
    # temperature of 0 means greedy, but it's not always supported by the server
    # so setting explicit greedy parameters instead
    if request["temperature"] == 0:
        request["temperature"] = 1.0
        request["top_k"] = 1
        request["top_p"] = 1.0


def postprocess_output(outputs: list[dict], stop_phrases: list[str]):
    """Post-processes the outputs of the model."""
    for output in outputs:
        output['generation'] = remove_stop_phrases(output['generation'], stop_phrases)


class BaseModel(abc.ABC):
    """Base model class for handling requests to the inference server.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self.server_host = host
        self.server_port = port
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            self.requests_lib = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
        else:
            self.requests_lib = requests

    @abc.abstractmethod
    def generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        pass


class TensorRTLLMModel(BaseModel):
    """Note that the current implementation supports inflight-batching so
    to make the most use of it, you should submit a large number of prompts
    at the same time.

    A good default value is 16-32 times bigger than the model's max batch size.
    """

    def generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
        }
        preprocess_request(request)

        generation_ids = []

        for prompt in prompts:
            request["prompt"] = prompt
            generation_ids.append(
                self.requests_lib.put(
                    url="http://{}:{}/start_generation".format(self.server_host, self.server_port),
                    data=json.dumps(request),
                    headers={"Content-Type": "application/json"},
                ).json()
            )

        outputs = [None] * len(generation_ids)
        finished_count = 0
        while finished_count < len(generation_ids):
            time.sleep(0.1)
            for pos, generation_id in enumerate(generation_ids):
                if outputs[pos] is not None:
                    continue
                result = self.requests_lib.put(
                    url="http://{}:{}/get_result".format(self.server_host, self.server_port),
                    data=json.dumps({'generation_id': generation_id}),
                    headers={"Content-Type": "application/json"},
                ).json()
                if result is not None:
                    finished_count += 1
                    outputs[pos] = {'generation': result}
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs


class NemoModel(BaseModel):
    def generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
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
        }
        preprocess_request(request)
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        outputs = [None] * len(generations['sentences'])
        for idx, generation in enumerate(generations['sentences']):
            outputs[idx] = {'generation': generation[len(prompts[idx]) :]}
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs


# TODO: this is broken
class OpenAIModel(BaseModel):
    def __init__(
        self,
        model,
        api_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if stop_phrases is None:
            stop_phrases = []
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI, please set it to default value `0`.")

        outputs = []
        for input_dict in input_dicts:
            response = self._send_request(
                prompt=prompt,
                input_dict=input_dict,
                tokens_to_generate=tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_phrases=stop_phrases,
            )
            outputs.append({'generation': response})

        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)

        return outputs

    def _send_request(
        self,
        prompt: str,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ) -> str:
        messages = prompt.build_structured(input_dict)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=tokens_to_generate,
            presence_penalty=repetition_penalty,
            seed=random_seed,
            stop=stop_phrases,
            messages=messages,
        ).choices[0]
        output = response.message.content
        # adding back stop words
        if response.finish_reason == "stop":
            output += response.stop_reason
        return output


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ssh_server and self.ssh_key_path:
            raise NotImplementedError("SSH tunnelling is not implemented for vLLM model.")

        self.server_type = "openai"
        self.oai_client = None
        self.oai_client = self.prepare_openai(self.server_host, self.server_port)  # type: openai.OpenAI

        self.model_name_server = self.get_model_name_from_server()
        self.model = self.model_name_server

        LOG.info("Model hosted by %s server: %s", self.server_type, self.model)

    def generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if stop_phrases is None:
            stop_phrases = []
        request = {
            'prompt': prompts,
            'max_tokens': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'num_generations': 1,  # VLLM provides 1 generation per prompt, duplicate prompts if you want more
            'stop': stop_phrases,
            'echo': False,
            'repetition_penalty': repetition_penalty,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'logprobs': None,
            'logit_bias': None,
            'seed': random_seed,
        }
        preprocess_request(request)
        outputs = [{'generation': output} for output in self.prompt_api(**request, parse_response=True)]
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs

    def prompt_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 1,
        num_generations: int = 1,
        stop=None,
        echo: bool = False,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: int = None,
        logit_bias: dict = None,
        seed: int = None,
        parse_response: bool = True,
    ) -> Union[list[str], "openai.types.Completion"]:
        if top_k == 0:
            top_k = 1

        # Process top_k
        extra_body = {
            "extra_body": {
                "top_k": 1,
                "repetition_penalty": repetition_penalty,
                "spaces_between_special_tokens": False,
            }
        }
        response = self.oai_client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=num_generations,
            stream=False,
            stop=stop,
            echo=echo,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            seed=seed,
            **extra_body,
        )

        if parse_response:
            response = self.parse_openai_response(response)

        return response

    @classmethod
    def parse_openai_response(cls, response: "openai.types.Completion") -> list[str]:
        responses = []
        if not isinstance(response, list):
            response = [response]

        for resp in response:
            for choice in resp.choices:
                output = choice.text
                # adding back stop words
                if choice.finish_reason == "stop":
                    output += choice.stop_reason
                responses.append(output)
        return responses

    @staticmethod
    def prepare_openai(host: str, port: str = "5000") -> "OpenAI":
        import openai

        # Update global config of openai
        openai.api_key = "EMPTY"
        openai.base_url = f"http://{host}:{port}/v1"

        # Create local client with no timeout
        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1", timeout=None)
        return client

    def get_model_name_from_server(self) -> str:
        model_list = self.oai_client.models.list()
        model_name = model_list.data[0].id
        return model_name


models = {
    'tensorrt_llm': TensorRTLLMModel,
    'nemo': NemoModel,
    # 'openai': OpenAIModel,
    'vllm': VLLMModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)
