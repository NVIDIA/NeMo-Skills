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

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
from openai import BadRequestError

from nemo_skills.utils import get_logger_name

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _ensure_server_ready(self) -> None:
        """
        VLLM/SGLANG/TensorRT-LLM servers need readiness checking as they take time to start up.
        """
        if not self.model:
            # No model name provided, discover it from server (this also checks readiness)
            self.model = self.get_model_name_from_server()
        else:
            # Model name provided, but still check server readiness
            self._wait_for_server_ready()
    
    def get_model_name_from_server(self, timeout: int = 300, retry_interval: int = 5) -> str:
        """
        Get the model name from VLLM/SGLANG server using OpenAI-compatible API.
        
        Args:
            timeout: Maximum time to wait for server readiness (seconds)
            retry_interval: Time between retries (seconds)
            
        Returns:
            str: The model name from the server
            
        Raises:
            Exception: If server is not ready after timeout period or no model found
        """
        # If model name is already provided, do a quick readiness check and return it
        if self.model:
            LOG.info("Waiting for server to be ready at %s:%s...", self.server_host, self.server_port)
            self._wait_for_vllm_server_ready(timeout, retry_interval)
            return self.model
            
        # Otherwise, try to get model name from server using OpenAI-compatible API
        LOG.info("Waiting for server to be ready at %s:%s...", self.server_host, self.server_port)
        return self._discover_model_from_vllm_server(timeout, retry_interval)
    
    def _wait_for_vllm_server_ready(self, timeout: int = 300, retry_interval: int = 5):
        """
        Wait for VLLM/SGLANG server to be ready using OpenAI-compatible /v1/models endpoint.
        """
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            try:
                if self._check_vllm_server_ready():
                    LOG.info("Server is ready!")
                    return
                    
            except Exception as e:
                # Log the specific error for debugging
                if attempt == 1:
                    LOG.debug("Server readiness check failed: %s", e)
                
            # Log progress every 30 seconds
            if attempt % 6 == 0:  # Log every 30 seconds (6 * 5 second intervals)
                LOG.info("Still waiting for server... (attempt %d/60)", attempt)
            
            time.sleep(retry_interval)
        
        # Timeout reached
        LOG.error("Server not ready after %d seconds", timeout)
        raise Exception(f"Server not ready after {timeout} seconds")
    
    def _check_vllm_server_ready(self) -> bool:
        """Check VLLM/SGLANG server readiness using OpenAI-compatible /v1/models endpoint."""
        import openai
        
        # Create a temporary OpenAI client for checking
        client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0  # Don't retry, we handle retries ourselves
        )
        
        model_list = client.models.list()
        return model_list.data is not None and len(model_list.data) > 0
    
    def _discover_model_from_vllm_server(self, timeout: int = 300, retry_interval: int = 5) -> str:
        """
        Discover the model name from VLLM/SGLANG server using OpenAI-compatible API.
        """
        import openai
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            try:
                # Try to get the model list from the server using OpenAI-compatible API
                client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    max_retries=0
                )
                
                model_list = client.models.list()
                if model_list.data:
                    model_name = model_list.data[0].id
                    LOG.info("Server is ready! Found model: %s", model_name)
                    return model_name
                else:
                    LOG.warning("Server responded but no models found")
                    
            except openai.NotFoundError:
                # 404 error - server might not be fully ready yet
                pass
            except (openai.APIConnectionError, ConnectionError):
                # Connection error - server not ready yet
                pass
            except Exception as e:
                # Other errors - log and continue
                LOG.debug("Error discovering model: %s", e)
            
            # Log progress every 30 seconds
            if attempt % 6 == 0:
                LOG.info("Still waiting for server... (attempt %d/60)", attempt)
            
            time.sleep(retry_interval)
        
        # Timeout reached
        LOG.error("Server not ready after %d seconds", timeout)
        raise Exception(f"Server not ready after {timeout} seconds")
    
    def _wait_for_server_ready(self, timeout: int = 300, retry_interval: int = 5):
        """Override base class to use VLLM-specific readiness check."""
        self._wait_for_vllm_server_ready(timeout, retry_interval)

    def _build_request_body(self, top_k, min_p, repetition_penalty, extra_body: dict = None):
        full_extra_body = {
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "spaces_between_special_tokens": False,
        }

        if top_k > 0:
            full_extra_body["top_k"] = top_k

        if extra_body:
            full_extra_body.update(extra_body)

        return full_extra_body

    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
    ) -> dict:
        assert reasoning_effort is None, "reasoning_effort is not supported for text completion requests"
        assert tools is None, "tools are not supported for text completion requests"
        return {
            "prompt": prompt,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs,
            "stream": stream,
            "echo": False,
            "skip_special_tokens": False,
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
        }

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict = None,
    ) -> dict:
        request = {
            "messages": messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs is not None,
            "top_logprobs": top_logprobs,
            "n": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": stream,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
            "tools": tools,
        }
        if reasoning_effort:
            request["allowed_openai_params"] = ["reasoning_effort"]
            request["reasoning_effort"] = reasoning_effort
        return request
