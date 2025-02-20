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
import uuid
import random
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
import requests
from openai import DefaultHttpxClient

LOG = logging.getLogger(__name__)


def trim_after_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


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
            # TODO: switch to httpx
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=1500, pool_connections=1500, max_retries=3)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self.requests_lib = session

        self.gen_id_to_params = {}
        self.gen_id_to_future = {}

        self.executor = ThreadPoolExecutor(max_workers=1024)  # is this too much?

    @abc.abstractmethod
    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int | list[int],
        temperature: float | list[float],
        top_p: float | list[float],
        top_k: int | list[int],
        min_p: float | list[float],
        repetition_penalty: float | list[float],
        random_seed: int | list[int],
        stop_phrases: list[str] | list[list[str]] | None,
    ) -> dict:
        """If the engine supports inflight-batching of requests, you only need to define this method.

        We will call it in threads on the list of prompts.
        """
        pass

    def preprocess_request(self, request: dict):
        """Just a small utility to pre-process some of the parameters of request."""
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if request["temperature"] == 0:
            request["temperature"] = 1.0
            request["top_k"] = 1
            request["top_p"] = 1.0

    def generate_async(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        """Returns a list of generation ids that can be later queried with get_generation calls."""
        kwargs = {
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
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

        gen_ids = []
        for request_idx in range(len(prompts)):
            # Prepare request
            request = {key: value[request_idx] for key, value in kwargs.items()}
            request['prompt'] = prompts[request_idx]
            self.preprocess_request(request)

            # Generate a unique generation ID
            gen_id = str(uuid.uuid4())
            gen_ids.append(gen_id)

            # Update global dictionaries tracking the progress of generations
            self.gen_id_to_future[gen_id] = self.executor.submit(self._generate_single, **request)
            
            self.gen_id_to_params[gen_id] = (kwargs["stop_phrases"][request_idx], remove_stop_phrases)

        return gen_ids

    def get_generations(self, generation_ids: list[str]) -> list[dict]:
        generations = []
        for generation_id in generation_ids:
            if generation_id not in self.gen_id_to_future:
                raise ValueError(f"Generation id {generation_id} not found.")

            stop_phrases, remove_stop_phrases = self.gen_id_to_params[generation_id]
            future = self.gen_id_to_future[generation_id]
            if not future.done():
                output = {'generation': None}
            else:
                output = future.result()
                del self.gen_id_to_future[generation_id]
                del self.gen_id_to_params[generation_id]

            if remove_stop_phrases:
                if output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        generation_ids = self.generate_async(
            prompts=prompts,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            remove_stop_phrases=remove_stop_phrases,
        )
        all_generations = [None] * len(prompts)
        while True:
            remaining_ids = [generation_id for generation_id in generation_ids if generation_id is not None]
            if len(remaining_ids) == 0:
                break
            remaining_positions = [
                idx for idx, generation_id in enumerate(generation_ids) if generation_id is not None
            ]
            generations = self.get_generations(remaining_ids)
            for gen_pos, gen_dict in zip(remaining_positions, generations):
                if gen_dict['generation'] is not None:  # will be None until done
                    generation_ids[gen_pos] = None
                    all_generations[gen_pos] = gen_dict

            time.sleep(1)

        return all_generations


class TRTLLMModel(BaseModel):
    """Note that the current implementation supports inflight-batching so
    to make the most use of it, you should submit a large number of prompts
    at the same time.

    A good default value is 16-32 times bigger than the model's max batch size.
    """

    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
    ) -> list[dict]:
        if isinstance(prompt, dict):
            raise NotImplementedError("trtllm server does not support OpenAI \"messages\" as prompt.")

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
        }
        output_dict = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        return output_dict

    # TODO: DRY
    def _generate_single_async(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
    ) -> list[dict]:
        if isinstance(prompt, dict):
            raise NotImplementedError("trtllm server does not support OpenAI \"messages\" as prompt.")

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
        }
        output_dict = self.requests_lib.put(
            url="http://{}:{}/generate_async".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()

        return output_dict['generation_id']

    def generate_async(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
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

        futures = []
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            for request_idx in range(len(prompts)):
                request = {key: value[request_idx] for key, value in kwargs.items()}
                request['prompt'] = prompts[request_idx]
                self.preprocess_request(request)
                futures.append(executor.submit(self._generate_single_async, **request))
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


class NemoModel(BaseModel):
    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int | list[int] = 512,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
    ) -> list[dict]:
        """If the engine supports inflight-batching of requests, you only need to define this method.

        We will call it in threads on the list of prompts.
        """
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")
        if isinstance(prompt, dict):
            raise NotImplementedError("NeMo server does not support OpenAI \"messages\" as prompt.")
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
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")

        # we are overriding generate directly, since nemo doesn't support inflight batching
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


class OpenAIModel(BaseModel):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        model=None,
        base_url=None,
        api_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if model is None:
            model = os.getenv("NEMO_SKILLS_OPENAI_MODEL")
            if model is None:
                raise ValueError("model argument is required for OpenAI model.")

        if base_url is None:
            # if not provided, we assume it's served on host/port
            base_url = os.getenv("NEMO_SKILLS_OPENAI_BASE_URL", f"http://{host}:{port}/v1")

        if api_key is None:
            if base_url is not None and 'api.nvidia.com' in base_url:
                api_key = os.getenv("NVIDIA_API_KEY", api_key)
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY is required for Nvidia-hosted models.")
            elif base_url is not None and 'api.openai.com' in base_url:
                api_key = os.getenv("OPENAI_API_KEY", api_key)
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI models.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        if self.model == "model":  # that's a placeholder, so trying to find real name
            self.model = self.get_model_name_from_server()

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

    def preprocess_request(self, request: dict):
        """OpenAI doesn't support top-k, so not making any changes here."""
        pass

    def _generate_single(
        self,
        prompt: dict,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
    ) -> str:
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to default value `0`.")
        if min_p > 0:
            ValueError("`min_p` is not supported by OpenAI API, please set it to default value `0`.")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=1,
                # top_p=top_p,
                max_completion_tokens=tokens_to_generate,
                presence_penalty=repetition_penalty,
                seed=random_seed,
                stop=stop_phrases,
                messages=prompt,
            )
            response = response.choices[0]
        except openai.BadRequestError as e:
            # this likely only works for Nvidia-hosted models
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
                    messages=prompt,
                ).choices[0]
            else:
                raise
        except AttributeError:
            # sometimes response is a string?
            LOG.error("Unexpected response from OpenAI API: %s", response)
            raise

        output = response.message.content
        return {'generation': output}

    def get_model_name_from_server(self):
        model_list = self.client.models.list()
        # TODO: this is a bit hacky, but will go away when we switch to a unified openai api for all models
        assert len(model_list.data) == 1, "Unexpected number of models returned by OpenAI API."
        model_name = model_list.data[0].id
        return model_name


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: move this to base model?
        self._tunnel = None
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel

            if '@' in self.ssh_server:
                ssh_username, ssh_server = self.ssh_server.split('@')
            else:
                ssh_server = self.ssh_server
                ssh_username = None

            self._tunnel = sshtunnel.SSHTunnelForwarder(
                (ssh_server, 22),
                ssh_username=ssh_username,
                ssh_pkey=self.ssh_key_path,
                remote_bind_address=(self.server_host, int(self.server_port)),
            )
            self._tunnel.start()
            # Use localhost with tunneled port for OpenAI client
            # This way all traffic to server_host:server_port goes through SSH tunnel
            self.server_host = '127.0.0.1'
            self.server_port = str(self._tunnel.local_bind_port)

        http_client = DefaultHttpxClient(
            limits=httpx.Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=httpx.HTTPTransport(retries=3),
        )

        self.oai_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.server_host}:{self.server_port}/v1",
            timeout=None,
            http_client=http_client,
        )

        self.model_name_server = self.get_model_name_from_server()
        self.model = self.model_name_server

    def __del__(self):
        if self._tunnel:
            self._tunnel.stop()


    def response_completion(self, prompt, tokens_to_generate=100, temperature=0.7, top_p=1.0, 
                            random_seed=None, stop_phrases=None, top_k=None, min_p=None, 
                            repetition_penalty=1.2):
        logit_bias = {
            14190: 10,  # " Wait"
            3983: 10, 582: 5, 1184: 5, 311: 5, 1779: 5, 421: 5, 419: 5, 4278: 5,
            92014: 10,
            # 10061: 10, 752: 5, 1430: 5, 2441: 5, 5486: 5,
            # 40: 10, 1184: 5, 311: 5, 1990: 5, 15934: 5,
            3983: 10, 1077: 5, 752: 5, 1779: 5, 1549: 5,
            10061: 10, 752: 5, 1744: 5, 1549: 5
        }


        return self.oai_client.completions.create(
            model=self.model,
            prompt=[prompt],
            max_tokens=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            seed=random_seed,
            stop=stop_phrases,
            echo=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=None,
            logit_bias=logit_bias,
            n=1,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "spaces_between_special_tokens": False,
            },
        )

    
    
    
    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
    ) -> dict:
        if isinstance(prompt, dict):
            raise NotImplementedError("TODO: need to add this support, but not implemented yet.")
        stop_phrases = stop_phrases or []

        if top_k == 0:
            top_k = -1

        final_output = ""
        ignore_strs = [" Wait", " But we need to check if this works", " Alternatively", " Let me try another approach", " I need to double-check", " But let me check again"
                       , " Let me think again"]
        ignore_str = " Wait"
        
        try_times = 0
        tokens_for_final_answer = 0
        max_tokens_thinking_tmp = tokens_to_generate - tokens_for_final_answer
        total_generated_tokens = 0
        num_generated_tokens = 0
        for i in range(try_times):
            max_tokens_thinking_tmp -= num_generated_tokens
            if max_tokens_thinking_tmp > 0:
                response = self.response_completion(prompt, max_tokens_thinking_tmp, temperature, top_p, random_seed, stop_phrases, top_k, min_p, repetition_penalty)
                output, num_generated_tokens = self.parse_openai_response(response)
                total_generated_tokens += num_generated_tokens
                ignore_str = random.choice(ignore_strs)
                prompt += (output + ignore_str)
                if i == try_times - 1:
                    final_output += output
                else:
                    final_output += (output + ignore_str)
        
        response = self.response_completion(prompt, max_tokens_thinking_tmp, temperature, top_p, random_seed, stop_phrases, top_k, min_p, repetition_penalty)
        final_output, num_generated_tokens = self.parse_openai_response(response)
                
        return {'generation': final_output, 'num_generated_tokens': num_generated_tokens}

    @classmethod
    def parse_openai_response(cls, response: "openai.types.Completion") -> tuple[str, int]:
        assert not isinstance(response, list)
        assert len(response.choices) == 1
        choice = response.choices[0]
        output = choice.text
        # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop
        num_generated_tokens = response.usage.completion_tokens
        return output, num_generated_tokens

    def get_model_name_from_server(self):
        model_list = self.oai_client.models.list()
        model_name = model_list.data[0].id
        return model_name


models = {
    'trtllm': TRTLLMModel,
    'nemo': NemoModel,
    'openai': OpenAIModel,
    'vllm': VLLMModel,
    'sglang': VLLMModel,  # interface is the same
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)
