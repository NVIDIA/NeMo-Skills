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
import asyncio
import logging
import os
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
import requests
from openai import DefaultHttpxClient, Stream

# TODO: Remove this once added to the docker image
import subprocess
subprocess.check_call(["pip", "install", "litellm==1.71.1"])
import litellm

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

    def _generate_single(
        self, *args, **kwargs,
    ) -> dict:
        """If the engine supports inflight-batching of requests, you only need to define this method.
        """
        raise NotImplementedError("This method should be implemented by the child class")

    def preprocess_request(self, request: dict):
        """Just a small utility to pre-process some of the parameters of request."""
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if request["temperature"] == 0:
            request["temperature"] = 1.0
            request["top_k"] = 1
            request["top_p"] = 1.0

    def generate_sync(
        self,
        prompt: str | list,
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
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        # Prepare the request parameters
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
            'include_response': include_response,
            'extra_body': extra_body,
        }
        if tools is not None:
            kwargs['tools'] = tools

        # Create the request for the single prompt
        request = kwargs.copy()
        request['prompt'] = prompt
        self.preprocess_request(request)

        # Generate directly using _generate_single
        output = self._generate_single(**request)

        # Apply stop phrase removal if needed
        if remove_stop_phrases and isinstance(output, dict) and output.get('generation') is not None:
            output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

        # Remove logprobs if not requested
        if top_logprobs is None:
            output.pop('tokens', None)
            output.pop('logprobs', None)

        return output

    async def generate_asyncio(self, prompt, *args, **kwargs) -> dict:
        # Configure the executor for the current event loop
        loop = asyncio.get_running_loop()
        if not hasattr(loop, '_nemo_skills_executor_configured'):
            loop.set_default_executor(ThreadPoolExecutor(max_workers=2048))
            loop._nemo_skills_executor_configured = True

        result = await asyncio.to_thread(self.generate_sync, prompt, *args, **kwargs)
        return result
    
class OpenAIAPIModel(BaseModel):
    """
    Base class for models using an OpenAI-compatible API.
    Handles client setup, SSH tunneling, and a unified generation flow with generation tracking.
    """
    # Litellm provider name
    MODEL_PROVIDER = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str = "EMPTY",
        base_url: str | None = None,
        max_retries: int = 3,
        use_v1_endpoint: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

        assert model is not None, "model is required"
        model_litellm = f"{self.MODEL_PROVIDER}/{model}"
        # Passed to litellm every time we call it
        self.litellm_kwargs = dict(
            model=model_litellm,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
        )
        httpx_limits = httpx.Limits(
            max_keepalive_connections=1500, max_connections=1500
        )
        litellm.client_session = httpx.Client(limits=httpx_limits)
        litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)


    def __del__(self):
        if self._tunnel:
            self._tunnel.stop()

    @abc.abstractmethod
    def _build_chat_request_params(self, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def _build_completion_request_params(self, **kwargs) -> dict:
        pass

    async def generate_asyncio(
        self,
        prompt: str | list,
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        top_logprobs: int | list[int] | None = None,
        timeout: float | int | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | list[int] | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> Union[dict, Stream, tuple[str, Union[dict, Stream]]]:
        """Native async version of generate for single prompt."""
        # Prepare request parameters similar to generate_async
        if timeout is None:
            # litellm uses 10min as default None
            # So we change it to 10000 seconds for consistency with other clients
            timeout = 10000 
        if top_k == 0:
            top_k = -1 # litellm doesn't support top_k=0
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
            'include_response': include_response,
            'extra_body': extra_body,
        }
        if tools is not None:
            kwargs['tools'] = tools

        request = kwargs.copy()
        request['prompt'] = prompt
        self.preprocess_request(request)

        result = await self._generate_single_async(**request)
        
        # Apply stop phrase removal if needed
        if remove_stop_phrases and isinstance(result, dict) and result.get('generation') is not None:
            from .utils import trim_after_stop_phrases
            result['generation'] = trim_after_stop_phrases(result['generation'], stop_phrases)

        return result
    
    def generate_sync(self, prompt, *args, **kwargs) -> dict:
        return asyncio.run(self.generate_asyncio(prompt, *args, **kwargs))

    async def _generate_single_async(
        self,
        prompt: str | list,
        stream: bool = False,
        include_response: bool = False,
        **kwargs,
    ) -> Union[dict, Stream, tuple[str, Union[dict, Stream]]]:
        if isinstance(prompt, list):
            request_params = self._build_chat_request_params(messages=prompt, stream=stream, **kwargs)
            response = await litellm.acompletion(**request_params, **self.litellm_kwargs)
            if stream:
                result = self._stream_chat_chunks_async(response)
            else:
                result = self._parse_chat_completion_response(response, include_response=include_response)

        elif isinstance(prompt, str):
            request_params = self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
            response = await litellm.atext_completion(**request_params, **self.litellm_kwargs)
            if stream:
                result = self._stream_completion_chunks_async(response)
            else:
                result = self._parse_completion_response(response, include_response=include_response)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        return result

    def _parse_completion_response(self, response: "openai.types.Completion", include_response: bool = False) -> dict:
        choice = response.choices[0]
        output = choice.text
        if output is None:
            output = ""

        # In some cases, the stop reason is not included in the text, so we add it back
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop

        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if getattr(choice, 'logprobs', None):
            result['logprobs'] = choice.logprobs.token_logprobs
            result['tokens'] = choice.logprobs.tokens
            result['top_logprobs'] = choice.logprobs.top_logprobs
        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason

        if include_response:
            result["response"] = response

        return result

    def _parse_chat_completion_response(self, response, include_response: bool = False) -> dict:
        choice = response.choices[0]
        output = choice.message.content
        if output is None:
            output = ""
        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if getattr(choice, 'logprobs', None) and choice.logprobs.content:
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
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            result["tool_calls"] = choice.message.tool_calls
        if include_response:
            result["response"] = response

        return result

    async def _stream_completion_chunks_async(self, response):
        """Async version of stream completion chunks."""
        emitted_so_far = []
        async for chunk in response:
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

    async def _stream_chat_chunks_async(self, response):
        """Async version of stream chat chunks."""
        async for chunk in response:
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

