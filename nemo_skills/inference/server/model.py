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
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import openai
import requests

LOG = logging.getLogger(__name__)


def trim_after_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


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
        output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)


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
        self.ssh_server = ssh_server
        self.ssh_key_path = ssh_key_path
        if ssh_server is None:
            self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
        if ssh_key_path is None:
            self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            self.requests_lib = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
        else:
            self.requests_lib = requests

    @abc.abstractmethod
    def generate(
        self,
        prompts: list[str | dict],
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


class TRTLLMModel(BaseModel):
    """Note that the current implementation supports inflight-batching so
    to make the most use of it, you should submit a large number of prompts
    at the same time.

    A good default value is 16-32 times bigger than the model's max batch size.
    """

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if isinstance(prompts[0], dict):
            raise NotImplementedError("trtllm server does not support OpenAI \"messages\" as prompt.")
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
        last_time = time.time()
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
                    last_time = time.time()
            # a hack to make sure we never hang indefinitely and
            # always abort the job if something is stuck in trt engine
            if time.time() - last_time > 300:
                raise RuntimeError("TRTLLM server is stuck, aborting the job. Please report this!")
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs


class NemoModel(BaseModel):
    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
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
            # when the prompt starts from special tokens like bos, nemo will remove them,
            # so we need this hack to find where to start the cut
            begin_idx = 0
            while begin_idx < len(prompts[idx]) and not prompts[idx][begin_idx:].startswith(generation[:20]):
                begin_idx += 1
            outputs[idx] = {'generation': generation[(len(prompts[idx]) - begin_idx) :]}

        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)
        return outputs


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model=None,
        base_url=None,
        api_key=None,
        max_parallel_requests: int = 100,  # can adjust to avoid rate-limiting
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if model is None:
            model = os.getenv("NEMO_SKILLS_OPENAI_MODEL")
            if model is None:
                raise ValueError("model argument is required for OpenAI model.")

        if base_url is None:
            base_url = os.getenv("NEMO_SKILLS_OPENAI_BASE_URL")

        if api_key is None:
            if base_url is not None and 'api.nvidia.com' in base_url:
                api_key = os.getenv("NVIDIA_API_KEY", api_key)
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY is required for Nvidia-hosted models.")
            else:
                api_key = os.getenv("OPENAI_API_KEY", api_key)
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI models.")

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_parallel_requests = max_parallel_requests

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        reduce_generation_tokens_if_error: bool = True,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if isinstance(prompts[0], str):
            raise NotImplementedError("OpenAI server requires \"messages\" dicts as prompt.")
        if stop_phrases is None:
            stop_phrases = []
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to default value `0`.")

        futures = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            for prompt in prompts:
                futures.append(
                    executor.submit(
                        self._send_request,
                        prompt=prompt,
                        tokens_to_generate=tokens_to_generate,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        random_seed=random_seed,
                        stop_phrases=stop_phrases,
                        reduce_generation_tokens_if_error=reduce_generation_tokens_if_error,
                    )
                )

        outputs = [{'generation': future.result()} for future in futures]
        if remove_stop_phrases:
            postprocess_output(outputs, stop_phrases)

        return outputs

    def batch_generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
    ) -> list[dict]:
        # only supported by the OpenAI endpoint!
        if stop_phrases is None:
            stop_phrases = []
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to default value `0`.")

        # preparing the requests jsonl file
        with open("requests.jsonl", "wt", encoding='utf-8') as fout:
            for idx, prompt in enumerate(prompts):
                fout.write(
                    json.dumps(
                        {
                            "custom_id": f"{idx}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model,
                                "messages": prompt,
                                "max_tokens": tokens_to_generate,
                                "temperature": temperature,
                                "top_p": top_p,
                                "presence_penalty": repetition_penalty,
                                "seed": random_seed,
                                "stop": stop_phrases,
                            },
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

    def _send_request(
        self,
        prompt: str,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
        reduce_generation_tokens_if_error: bool = True,
    ) -> str:
        import openai

        messages = prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=tokens_to_generate,
                presence_penalty=repetition_penalty,
                seed=random_seed,
                stop=stop_phrases,
                messages=messages,
            )
            response = response.choices[0]
        except openai.BadRequestError as e:
            # this likely only works for Nvidia-hosted models
            if not reduce_generation_tokens_if_error:
                raise
            msg = e.body['detail']
            # expected message:
            # This model's maximum context length is N tokens.
            # However, you requested X tokens (Y in the messages, Z in the completion).
            # Please reduce the length of the messages or completion.
            if msg.startswith("This model's maximum context length is"):
                numbers = re.findall(r"\d+", msg)
                max_tokens = int(numbers[0]) - int(numbers[2])
                LOG.warning("Reached max tokens! Reducing the number of tokens to generate to %d", max_tokens)
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=repetition_penalty,
                    seed=random_seed,
                    stop=stop_phrases,
                    messages=messages,
                ).choices[0]
            else:
                raise
        except AttributeError:
            # sometimes response is a string?
            LOG.error("Unexpected response from OpenAI API: %s", response)
            raise

        output = response.message.content
        return output


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ssh_server and self.ssh_key_path:
            raise NotImplementedError("SSH tunnelling is not implemented for vLLM model.")

        self.oai_client = openai.OpenAI(
            api_key="EMPTY", base_url=f"http://{self.server_host}:{self.server_port}/v1", timeout=None
        )

        self.model_name_server = self.get_model_name_from_server()
        self.model = self.model_name_server

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if isinstance(prompts[0], dict):
            raise NotImplementedError("TODO: need to add this support, but not implemented yet.")
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
        top_k: int = -1,
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
    ) -> Union[list[str], openai.types.Completion]:
        if top_k == 0:
            top_k = -1

        # Process top_k
        extra_body = {
            "extra_body": {
                "top_k": top_k,
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
                # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
                if choice.finish_reason == "stop" and isinstance(choice.stop_reason, str):
                    output += choice.stop_reason
                responses.append(output)
        return responses

    def get_model_name_from_server(self):
        model_list = self.oai_client.models.list()
        model_name = model_list.data[0].id
        return model_name


models = {
    'trtllm': TRTLLMModel,
    'nemo': NemoModel,
    'openai': OpenAIModel,
    'vllm': VLLMModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)
