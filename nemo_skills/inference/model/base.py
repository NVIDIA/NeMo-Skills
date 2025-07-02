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

import abc
import logging
import os
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Union, Optional
import threading

import httpx
import openai
import requests
from openai import DefaultHttpxClient, Stream

from nemo_skills.utils import get_logger_name

from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))


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
        prompt: str | list,
        tokens_to_generate: int | list[int],
        temperature: float | list[float],
        top_p: float | list[float],
        top_k: int | list[int],
        min_p: float | list[float],
        repetition_penalty: float | list[float],
        random_seed: int | list[int],
        stop_phrases: list[str] | list[list[str]] | None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,
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
        prompts: list[str | list],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,
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
            'top_logprobs': top_logprobs,
            'timeout': timeout,
            'stream': stream,
            'reasoning_effort': reasoning_effort,
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
                if isinstance(output, dict) and output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations

    def generate(
        self,
        prompts: list[str | list],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,
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
            top_logprobs=top_logprobs,
            remove_stop_phrases=remove_stop_phrases,
            stream=stream,
            reasoning_effort=reasoning_effort,
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
                if isinstance(gen_dict, Generator) or gen_dict['generation'] is not None:  # will be None until done
                    generation_ids[gen_pos] = None
                    all_generations[gen_pos] = gen_dict
                    # trtllm always return these fields so we need to remove them if not requested
                    if isinstance(gen_dict, dict) and top_logprobs is None:
                        gen_dict.pop('tokens', None)
                        gen_dict.pop('logprobs', None)

            time.sleep(1)

        return all_generations


class OpenAIAPIModel(BaseModel):
    """
    Base class for models using an OpenAI-compatible API.
    Handles client setup, SSH tunneling, and a unified generation flow with generation tracking.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str = "EMPTY",
        base_url: str | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        use_v1_endpoint: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Track active generations with thread-safe operations
        self.active_generations: Dict[str, Dict[str, Any]] = {}
        self._generations_lock = threading.Lock()

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
            self.server_host = '127.0.0.1'
            self.server_port = str(self._tunnel.local_bind_port)

        if base_url is None:
            v1_suffix = "/v1" if use_v1_endpoint else ""
            base_url = f"http://{self.server_host}:{self.server_port}{v1_suffix}"

        http_client = DefaultHttpxClient(
            limits=httpx.Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=httpx.HTTPTransport(retries=3),
        )

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=None,
            http_client=http_client,
        )
        self.model = model or self.get_model_name_from_server()

    def __del__(self):
        if self._tunnel:
            self._tunnel.stop()
        # Clean up any remaining active generations
        self.cancel_all_generations()

    def get_model_name_from_server(self):
        model_list = self.client.models.list()
        if not model_list.data:
            raise ValueError("No models available on the server.")
        return model_list.data[0].id

    def _register_generation(self, gen_id: str, response: Any) -> None:
        """Register a new generation with the tracker."""
        with self._generations_lock:
            self.active_generations[gen_id] = {
                'response': response,
                'created_at': time.time()
            }

    def _unregister_generation(self, gen_id: str) -> Optional[Dict[str, Any]]:
        """Remove a generation from the tracker and return its info."""
        with self._generations_lock:
            return self.active_generations.pop(gen_id, None)

    def cancel_generation(self, gen_id: str) -> bool:
        """
        Cancel a specific generation by ID.
        Returns True if the generation was found and cancelled, False otherwise.
        """
        generation_info = self._unregister_generation(gen_id)
        if generation_info is None:
            return False

        generation_info['response'].close() 
        return True

    def cancel_all_generations(self) -> int:
        """
        Cancel all active generations.
        Returns the number of generations that were cancelled.
        """
        with self._generations_lock:
            generation_ids = list(self.active_generations.keys())
        
        cancelled_count = 0
        for gen_id in generation_ids:
            if self.cancel_generation(gen_id):
                cancelled_count += 1
        
        return cancelled_count

    def get_active_generation_count(self) -> int:
        """Return the number of currently active generations."""
        with self._generations_lock:
            return len(self.active_generations)

    def get_active_generation_ids(self) -> list[str]:
        """Return a list of all active generation IDs."""
        with self._generations_lock:
            return list(self.active_generations.keys())

    def _make_api_call(self, api_func, params, gen_id: str):
        retry_count = 0
        retry_delay = self.initial_retry_delay
        while True:
            try:
                response = api_func(**params)
                # Register the generation after successful API call, only for streaming responses
                is_streaming = params.get('stream', False)
                if is_streaming:
                    self._register_generation(gen_id, response)
                return response
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    LOG.error("Rate limit exceeded maximum retry attempts (%d). Giving up.", self.max_retries)
                    raise
                retry_after = getattr(e, 'retry_after', None)
                wait_time = float(retry_after) if retry_after is not None else retry_delay
                LOG.warning(
                    "Rate limit exceeded. Retrying in %.2f seconds... (Attempt %d/%d)",
                    wait_time,
                    retry_count,
                    self.max_retries,
                )
                time.sleep(wait_time)
                if retry_after is None:
                    retry_delay *= 2  # Exponential backoff if no header is provided
            except Exception as e:
                LOG.error("Unexpected error during API call: %s", str(e))
                raise

    @abc.abstractmethod
    def _build_chat_request_params(self, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def _build_completion_request_params(self, **kwargs) -> dict:
        pass

    def _generate_single(
        self,
        prompt: str | list,
        stream: bool = False,
        generation_id: Optional[str] = None,
        **kwargs,
    ) -> Union[dict, Stream, tuple[str, Union[dict, Stream]]]:
        """
        Generate a single response with optional generation tracking.
        
        Args:
            prompt: The input prompt (string or list of messages)
            stream: Whether to stream the response
            generation_id: Optional generation ID. If None, one will be generated.
            **kwargs: Additional parameters for the API call
            
        Returns:
            If generation_id is provided in kwargs, returns (gen_id, response)
            Otherwise returns just the response for backward compatibility
        """
        # Generate a unique ID for this generation
        gen_id = generation_id or str(uuid.uuid4())
        return_gen_id = generation_id is not None or kwargs.get('return_generation_id', False)
        
        try:
            if isinstance(prompt, list):
                request_params = self._build_chat_request_params(messages=prompt, stream=stream, **kwargs)
                response = self._make_api_call(self.client.chat.completions.create, request_params, gen_id)
                if stream:
                    result = self._stream_chat_chunks(response, gen_id)
                else:
                    result = self._parse_chat_completion_response(response)

            elif isinstance(prompt, str):
                request_params = self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
                response = self._make_api_call(self.client.completions.create, request_params, gen_id)
                if stream:
                    result = self._stream_completion_chunks(response, gen_id)
                else:
                    result = self._parse_completion_response(response)
            else:
                raise TypeError(f"Unsupported prompt type: {type(prompt)}")
            
            if return_gen_id:
                return gen_id, result
            else:
                return result
                
        except Exception as e:
            # Make sure to unregister the generation if an error occurs
            self._unregister_generation(gen_id)
            raise

    def _parse_completion_response(self, response: "openai.types.Completion") -> dict:
        choice = response.choices[0]
        output = choice.text

        # In some cases, the stop reason is not included in the text, so we add it back
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop

        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if choice.logprobs:
            result['logprobs'] = choice.logprobs.token_logprobs
            result['tokens'] = choice.logprobs.tokens
            result['top_logprobs'] = choice.logprobs.top_logprobs
        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason
        return result

    def _parse_chat_completion_response(self, response) -> dict:
        choice = response.choices[0]
        output = choice.message.content
        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if choice.logprobs and choice.logprobs.content:
            result['logprobs'] = [tok.logprob for tok in choice.logprobs.content]
            result['tokens'] = [tok.token for tok in choice.logprobs.content]
            result['top_logprobs'] = []
            for token_logprob in choice.logprobs.content:
                logprob = {entry.token: entry.logprob for entry in token_logprob.top_logprobs}
                if token_logprob.token not in logprob:
                    logprob[token_logprob.token] = token_logprob.logprob
                result['top_logprobs'].append(logprob)
        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason
        return result

    def _stream_completion_chunks(self, response, gen_id: str):
        """Stream completion chunks and automatically unregister when done."""
        try:
            emitted_so_far = []
            for chunk in response:
                cur_delta = chunk.choices[0].text
                emitted_so_far += [cur_delta]
                if cur_delta:
                    yield {"generation": cur_delta}
                # vllm variant
                stop_reason = getattr(chunk.choices[0], "stop_reason", None)
                # sglang variant
                matched_stop = getattr(chunk.choices[0], "matched_stop", None)
                # vllm variant - emit stop_reason as is and finish
                if stop_reason and isinstance(stop_reason, str):
                    yield {"generation": stop_reason}
                # sglang variant - emit only not-yet-sent part of matched_stop
                if matched_stop and isinstance(matched_stop, str):
                    remaining = matched_stop
                    # find the longest prefix of matched_stop that is already at
                    # the end of what we've emitted.
                    emitted_str = "".join(emitted_so_far)
                    max_len = min(len(emitted_str), len(matched_stop))
                    for i in range(max_len, 0, -1):
                        if emitted_str.endswith(matched_stop[:i]):
                            remaining = matched_stop[i:]
                            break
                    if remaining:
                        yield {"generation": remaining}
        finally:
            # Always unregister the generation when streaming is complete
            self._unregister_generation(gen_id)

    def _stream_chat_chunks(self, response, gen_id: str):
        """Stream chat chunks and automatically unregister when done."""
        try:
            for chunk in response:
                if hasattr(chunk.choices[0], "delta"):
                    cur_delta = chunk.choices[0].delta.content
                else:
                    cur_delta = chunk.choices[0].text

                finish_reason = getattr(chunk.choices[0], "finish_reason", None)
                result = {"generation": cur_delta}
                if finish_reason:
                    result["finish_reason"] = finish_reason
                    if not cur_delta:
                        result["generation"] = ""

                yield result
        finally:
            # Always unregister the generation when streaming is complete
            self._unregister_generation(gen_id)


class BaseRewardModel(abc.ABC):
    """Base model class for handling requests to the reward model inference server.

    Args:
        model_type: Reward model type
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
        model_type: str,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self.model_type = model_type
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

    def score(self, prompts: list[str]) -> list[dict]:
        pass
