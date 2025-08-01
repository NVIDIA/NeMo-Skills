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
try:
    import litellm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "litellm==1.71.1"])
    import litellm

from nemo_skills.utils import get_logger_name

from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))


class BaseModel:
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
    # Litellm provider name
    MODEL_PROVIDER = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str = "EMPTY",
        base_url: str | None = None,
        max_retries: int = 3,
        use_v1_endpoint: bool = True,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self._tunnel = None
        self.server_host = host
        self.server_port = port
        self.ssh_server = ssh_server
        self.ssh_key_path = ssh_key_path
        if ssh_server is None:
            self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
        if ssh_key_path is None:
            self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

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

    def _maybe_apply_stop_phrase_removal(self, result: dict, remove_stop_phrases: bool, stop_phrases: list[str] | list[list[str]] | None) -> None:
        if remove_stop_phrases and isinstance(result, dict) and result.get('generation') is not None:
            result['generation'] = trim_after_stop_phrases(result['generation'], stop_phrases)

    @abc.abstractmethod
    def _build_chat_request_params(self, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def _build_completion_request_params(self, **kwargs) -> dict:
        pass

    def _prepare_generation_params(
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
    ) -> dict:
        """Prepare generation parameters for both sync and async methods."""
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
        return request

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
        request = self._prepare_generation_params(
            prompt=prompt,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            top_logprobs=top_logprobs,
            timeout=timeout,
            remove_stop_phrases=remove_stop_phrases,
            stream=stream,
            reasoning_effort=reasoning_effort,
            tools=tools,
            include_response=include_response,
            extra_body=extra_body,
        )
        result = await self._generate_single_async(**request)
        self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)

        return result
    
    def generate_sync(
        self,
        *args,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
        **kwargs,
    ) -> Union[dict, Stream, tuple[str, Union[dict, Stream]]]:
        """
        Synchronous version of generate for single prompt.
        See generate_asyncio for full list of parameters.
        """
        request = self._prepare_generation_params(
            *args,
            stop_phrases=stop_phrases,
            remove_stop_phrases=remove_stop_phrases,
            **kwargs,
        )
        result = self._generate_single_sync(**request)
        self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
        return result

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
                result = self._parse_chat_completion_response(response, include_response=include_response, **kwargs)

        elif isinstance(prompt, str):
            request_params = self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
            response = await litellm.atext_completion(**request_params, **self.litellm_kwargs)
            if stream:
                result = self._stream_completion_chunks_async(response)
            else:
                result = self._parse_completion_response(response, include_response=include_response, **kwargs)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        return result

    def _generate_single_sync(
        self,
        prompt: str | list,
        stream: bool = False,
        include_response: bool = False,
        **kwargs,
    ) -> Union[dict, Stream, tuple[str, Union[dict, Stream]]]:
        """Synchronous version of _generate_single_async."""
        if isinstance(prompt, list):
            request_params = self._build_chat_request_params(messages=prompt, stream=stream, **kwargs)
            response = litellm.completion(**request_params, **self.litellm_kwargs)
            if stream:
                result = self._stream_chat_chunks_sync(response)
            else:
                result = self._parse_chat_completion_response(response, include_response=include_response, **kwargs)

        elif isinstance(prompt, str):
            request_params = self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
            response = litellm.text_completion(**request_params, **self.litellm_kwargs)
            if stream:
                result = self._stream_completion_chunks_sync(response)
            else:
                result = self._parse_completion_response(response, include_response=include_response, **kwargs)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        return result

    def _parse_completion_response(self, response: "openai.types.Completion", include_response: bool = False, **kwargs) -> dict:
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

    def _parse_chat_completion_response(self, response, include_response: bool = False, **kwargs) -> dict:
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

    def _process_completion_chunk(self, chunk, emitted_so_far: list):
        """Process a single completion chunk and return data to yield."""
        cur_delta = chunk.choices[0].text
        emitted_so_far.append(cur_delta)
        
        results_to_yield = []
        if cur_delta:
            results_to_yield.append({"generation": cur_delta})
        
        # vllm variant
        stop_reason = getattr(chunk.choices[0], "stop_reason", None)
        # sglang variant
        matched_stop = getattr(chunk.choices[0], "matched_stop", None)
        
        # vllm variant - emit stop_reason as is and finish
        if stop_reason and isinstance(stop_reason, str):
            results_to_yield.append({"generation": stop_reason})
        
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
                results_to_yield.append({"generation": remaining})
        
        return results_to_yield

    def _process_chat_chunk(self, chunk):
        """Process a single chat chunk and return data to yield."""
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

        return [result]

    def _stream_completion_chunks_sync(self, response):
        """Synchronous version of stream completion chunks."""
        emitted_so_far = []
        for chunk in response:
            results = self._process_completion_chunk(chunk, emitted_so_far)
            for result in results:
                yield result

    def _stream_chat_chunks_sync(self, response):
        """Synchronous version of stream chat chunks."""
        for chunk in response:
            results = self._process_chat_chunk(chunk)
            for result in results:
                yield result

    async def _stream_completion_chunks_async(self, response):
        """Async version of stream completion chunks."""
        emitted_so_far = []
        async for chunk in response:
            results = self._process_completion_chunk(chunk, emitted_so_far)
            for result in results:
                yield result

    async def _stream_chat_chunks_async(self, response):
        """Async version of stream chat chunks."""
        async for chunk in response:
            results = self._process_chat_chunk(chunk)
            for result in results:
                yield result

