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
from typing import Union

import httpx
import litellm
import openai
from openai import AsyncOpenAI, OpenAI

from nemo_skills.utils import get_logger_name

from .context_retry import ContextLimitRetryConfig, with_context_retry
from .utils import ServerTokenizer, WrapperAutoTokenizer, trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))


class BaseClientHandler:
    """Base client handler with clean public API"""

    def __init__(self, model_instance):
        self.model = model_instance
        self.defaults = None  # Will be set in subclasses
        self.setup_clients()  # Public method

    def setup_clients(self):
        """Public method for setting up API clients - override in subclasses"""
        pass

    def get_supported_params(self) -> set:
        """Public method to get supported parameters - override in subclasses"""
        raise NotImplementedError()

    def extract_and_validate_params(self, **kwargs) -> dict:
        """Public method for parameter extraction and validation"""
        supported = self.get_supported_params()

        # Let model filter/restrict parameters
        model_supported = self.model.get_supported_params()
        if model_supported:
            supported = supported.intersection(model_supported)

        # Check for unsupported parameters, but only for parameters that differ from defaults
        provided = set(kwargs.keys())
        unsupported_non_default = set()

        for param_name in provided:
            if param_name not in supported:
                if hasattr(self.defaults, param_name) and kwargs[param_name] == getattr(self.defaults, param_name):
                    continue  # Default value is allowed even if unsupported
                unsupported_non_default.add(param_name)

        if unsupported_non_default:
            raise ValueError(
                f"Unsupported parameters for {self.__class__.__name__}: {unsupported_non_default}. Supported: {sorted(supported)}"
            )

        # Extract with defaults - include all provided parameters, even unsupported ones if they're default values
        params = {}
        all_param_names = supported.union(provided)
        for param_name in all_param_names:
            if param_name in kwargs:
                params[param_name] = kwargs[param_name]
            else:
                params[param_name] = getattr(self.defaults, param_name)

        return params

    def build_request_structure(self, prompt, params: dict) -> dict:
        """Public method for building client-specific request structure"""
        if isinstance(prompt, list):
            return self.build_chat_request_structure(prompt, params)
        else:
            return self.build_completion_request_structure(prompt, params)

    def build_chat_request_structure(self, messages: list, params: dict) -> dict:
        """Public method for chat request structure - override in subclasses"""
        raise NotImplementedError()

    def build_completion_request_structure(self, prompt: str, params: dict) -> dict:
        """Public method for completion request structure - override in subclasses"""
        raise NotImplementedError()

    async def call_api_async(self, prompt, **kwargs):
        """Public method for async API calls"""
        # Extract and validate parameters
        params = self.extract_and_validate_params(**kwargs)

        # Build request using two-stage process
        request = self.build_request_structure(prompt, params)
        request = self.model.apply_model_specific_params(request, params)

        # Make the actual API call
        return await self.make_async_call(request, prompt)

    def call_api_sync(self, prompt, **kwargs):
        """Public method for sync API calls"""
        # Extract and validate parameters
        params = self.extract_and_validate_params(**kwargs)

        # Build request using two-stage process
        request = self.build_request_structure(prompt, params)
        request = self.model.apply_model_specific_params(request, params)

        # Make the actual API call
        return self.make_sync_call(request, prompt)

    async def make_async_call(self, request: dict, prompt):
        """Public method for making async API calls - override in subclasses"""
        raise NotImplementedError()

    def make_sync_call(self, request: dict, prompt):
        """Public method for making sync API calls - override in subclasses"""
        raise NotImplementedError()

    def parse_response(self, response, **kwargs) -> dict:
        """Public method for parsing responses - override in subclasses"""
        raise NotImplementedError()


class ChatCompletionHandler(BaseClientHandler):
    """Handler for chat completion and text completion APIs via litellm"""

    def __init__(self, model_instance):
        from .defaults import CHAT_COMPLETION_PARAMS, GenerationDefaults

        super().__init__(model_instance)
        self.defaults = GenerationDefaults()
        self.supported_params = CHAT_COMPLETION_PARAMS

    def get_supported_params(self) -> set:
        return self.supported_params

    def setup_clients(self):
        """Setup litellm configuration"""
        model_litellm = f"{self.model.MODEL_PROVIDER}/{self.model.model_name_or_path}"
        self.litellm_kwargs = dict(
            model=model_litellm,
            max_retries=getattr(self.model, "max_retries", 3),
            api_key=self.model.api_key,
            base_url=self.model.base_url,
        )
        # Setup litellm sessions
        httpx_limits = httpx.Limits(max_keepalive_connections=2048, max_connections=2048)
        litellm.client_session = httpx.Client(limits=httpx_limits)
        litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

    def build_chat_request_structure(self, messages: list, params: dict) -> dict:
        """Build chat completion request structure"""
        request = {
            "messages": messages,
            "max_tokens": params["tokens_to_generate"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "seed": params["random_seed"],
            "stop": params["stop_phrases"],
            "logprobs": params["top_logprobs"] is not None,
            "top_logprobs": params["top_logprobs"],
            "stream": params["stream"],
            "tools": params["tools"],
            "timeout": params["timeout"],
        }

        # Add non-standard parameters to extra_body
        extra_body = params.get("extra_body", {}).copy() if params.get("extra_body") else {}
        if params.get("top_k", -1) != -1:
            extra_body["top_k"] = params["top_k"]
        if params.get("min_p", 0.0) != 0.0:
            extra_body["min_p"] = params["min_p"]
        if params.get("repetition_penalty", 1.0) != 1.0:
            extra_body["repetition_penalty"] = params["repetition_penalty"]

        if extra_body:
            request["extra_body"] = extra_body

        return request

    def build_completion_request_structure(self, prompt: str, params: dict) -> dict:
        """Build text completion request structure"""
        request = {
            "prompt": prompt,
            "max_tokens": params["tokens_to_generate"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "seed": params["random_seed"],
            "stop": params["stop_phrases"],
            "logprobs": params["top_logprobs"],
            "stream": params["stream"],
            "timeout": params["timeout"],
        }

        # Add non-standard parameters to extra_body
        extra_body = params.get("extra_body", {}).copy() if params.get("extra_body") else {}
        if params.get("top_k", -1) != -1:
            extra_body["top_k"] = params["top_k"]
        if params.get("min_p", 0.0) != 0.0:
            extra_body["min_p"] = params["min_p"]
        if params.get("repetition_penalty", 1.0) != 1.0:
            extra_body["repetition_penalty"] = params["repetition_penalty"]

        if extra_body:
            request["extra_body"] = extra_body

        return request

    async def make_async_call(self, request: dict, prompt):
        """Make async API call via litellm"""
        if isinstance(prompt, list):
            return await litellm.acompletion(**request, **self.litellm_kwargs)
        else:
            return await litellm.atext_completion(**request, **self.litellm_kwargs)

    def make_sync_call(self, request: dict, prompt):
        """Make sync API call via litellm"""
        if isinstance(prompt, list):
            return litellm.completion(**request, **self.litellm_kwargs)
        else:
            return litellm.text_completion(**request, **self.litellm_kwargs)

    def parse_response(self, response, **kwargs) -> dict:
        """Parse response using existing BaseModel methods"""
        if hasattr(response, "choices") and hasattr(response.choices[0], "message"):
            return self.model.parse_chat_completion_response(response, **kwargs)
        else:
            return self.model.parse_completion_response(response, **kwargs)


class ResponsesHandler(BaseClientHandler):
    """Handler for responses API using direct OpenAI client"""

    def __init__(self, model_instance):
        from .defaults import RESPONSES_PARAMS, GenerationDefaults

        super().__init__(model_instance)
        self.defaults = GenerationDefaults()
        self.supported_params = RESPONSES_PARAMS

    def get_supported_params(self) -> set:
        return self.supported_params

    def setup_clients(self):
        """Setup OpenAI clients directly"""
        self.sync_client = OpenAI(base_url=self.model.base_url, api_key=self.model.api_key)
        self.async_client = AsyncOpenAI(base_url=self.model.base_url, api_key=self.model.api_key)

    def build_chat_request_structure(self, messages: list, params: dict) -> dict:
        """Build responses API request structure"""
        # Use proper list of dicts format for vLLM servers
        request = {
            "input": messages,
            "max_output_tokens": params["tokens_to_generate"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "stream": params["stream"],
        }

        # Only include tools if they are provided
        if params["tools"] is not None:
            request["tools"] = params["tools"]

        # Add non-standard parameters to extra_body (OpenAI responses API requirement)
        extra_body = {}
        if params["random_seed"] is not None:
            extra_body["seed"] = params["random_seed"]
        if params["reasoning_effort"] is not None:
            extra_body["reasoning_effort"] = params["reasoning_effort"]
        if params["timeout"] is not None:
            extra_body["timeout"] = params["timeout"]
        if params["stop_phrases"] is not None:
            extra_body["stop"] = params["stop_phrases"]
        if params["top_logprobs"] is not None:
            extra_body["top_logprobs"] = params["top_logprobs"]
        if params["top_k"] != -1:  # Only include if not default
            extra_body["top_k"] = params["top_k"]
        if params["min_p"] != 0.0:  # Only include if not default
            extra_body["min_p"] = params["min_p"]
        if params["repetition_penalty"] != 1.0:  # Only include if not default
            extra_body["repetition_penalty"] = params["repetition_penalty"]

        # Add any additional extra_body parameters
        if params["extra_body"]:
            extra_body.update(params["extra_body"])

        if extra_body:
            request["extra_body"] = extra_body

        return request

    def build_completion_request_structure(self, prompt: str, params: dict) -> dict:
        """Responses API doesn't support completion - raise error"""
        raise ValueError("ResponsesHandler only supports message lists, not string prompts")

    async def make_async_call(self, request: dict, prompt):
        """Make async call to responses API"""
        return await self.async_client.responses.create(model=self.model.model_name_or_path, **request)

    def make_sync_call(self, request: dict, prompt):
        """Make sync call to responses API"""
        return self.sync_client.responses.create(model=self.model.model_name_or_path, **request)

    def parse_response(self, response, **kwargs) -> dict:
        """Parse responses API response"""
        return self.model.parse_responses_response(response, **kwargs)


# Global client handler registry
CLIENT_HANDLERS = {
    "chat_completion": ChatCompletionHandler,
    "responses": ResponsesHandler,
    "completion": ChatCompletionHandler,  # Same handler, different method selection
}


class BaseModel:
    """Base model class for handling requests to the inference server.

    Args:
        model: str - Model name or path to use for inference.
        use_responses_api: bool = False - Whether to use responses API instead of chat completion API.
        tokenizer: str | None = None - Tokenizer to use for the model.
        api_key: str | None = None - API key for authentication.
        api_key_env_var: str | None = None - Environment variable name containing API key.
        base_url: str | None = None - Base URL for the API server.
        use_v1_endpoint: bool = True - Whether to use v1 endpoint format.
        host: str = '127.0.0.1' - Host of the inference server.
        port: str = '5000' - Port of the inference server.
        max_retries: int = 3 - Maximum number of retries for API calls.
        ssh_server: str | None = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: str | None = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
        enable_soft_fail: bool = False - Enable soft failure handling.
        context_limit_retry_strategy: str | None = None - Context limit retry strategy.
        num_special_tokens_budget: int = 100 - Budget for special tokens.
    """

    # Litellm provider name
    MODEL_PROVIDER = "openai"

    def __init__(
        self,
        model: str,
        use_responses_api: bool = False,
        tokenizer: str | None = None,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        base_url: str | None = None,
        use_v1_endpoint: bool = True,
        host: str = "127.0.0.1",
        port: str = "5000",
        max_retries: int = 3,
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
        # Context limit retry config variables
        enable_soft_fail: bool = False,
        context_limit_retry_strategy: str | None = None,
        num_special_tokens_budget: int = 100,
    ):
        # Common model properties
        self.model_name_or_path = model
        self.use_responses_api = use_responses_api
        self.max_retries = max_retries
        self.server_host = host
        self.server_port = port

        # SSH tunnel setup (general networking)
        self._setup_ssh_tunnel(ssh_server, ssh_key_path)

        # Base URL setup (general)
        self.base_url = self._setup_base_url(base_url, use_v1_endpoint)

        # API key resolution (general)
        self.api_key = self._resolve_api_key(api_key, api_key_env_var, base_url)

        # Context retry config (general)
        self.context_limit_retry_config = ContextLimitRetryConfig(
            enable_soft_fail=enable_soft_fail,
            strategy=context_limit_retry_strategy,
            num_special_tokens_budget=num_special_tokens_budget,
        )

        # Tokenizer setup (general)
        if enable_soft_fail:
            self.tokenizer = self._get_tokenizer(tokenizer)
        else:
            self.tokenizer = None

        # Initialize client handler LAST
        # Determine client type based on use_responses_api flag
        if use_responses_api:
            selected_client_type = "responses"
        else:
            selected_client_type = "chat_completion"

        if selected_client_type not in CLIENT_HANDLERS:
            raise ValueError(
                f"Unsupported client handler: {selected_client_type}. Available: {list(CLIENT_HANDLERS.keys())}"
            )

        handler_class = CLIENT_HANDLERS[selected_client_type]
        self.client_handler = handler_class(self)  # Public attribute

    def _setup_ssh_tunnel(self, ssh_server: str | None, ssh_key_path: str | None):
        """Setup SSH tunnel if needed"""
        self._tunnel = None
        self.ssh_server = ssh_server or os.getenv("NEMO_SKILLS_SSH_SERVER")
        self.ssh_key_path = ssh_key_path or os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel

            if "@" in self.ssh_server:
                ssh_username, ssh_server = self.ssh_server.split("@")
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
            self.server_host = "127.0.0.1"
            self.server_port = str(self._tunnel.local_bind_port)

    def _setup_base_url(self, base_url: str | None, use_v1_endpoint: bool) -> str:
        """Setup base URL for API calls"""
        if base_url is None:
            v1_suffix = "/v1" if use_v1_endpoint else ""
            return f"http://{self.server_host}:{self.server_port}{v1_suffix}"
        elif base_url == "":
            return None
        else:
            return base_url

    def _resolve_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        """Resolve API key from various sources"""
        resolved_key = self._get_api_key(api_key, api_key_env_var, base_url)
        if resolved_key is None:  # self-hosted models don't need the key
            resolved_key = "EMPTY"
        return resolved_key

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        if api_key:  # explicit cmd argument always takes precedence
            return api_key
        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"You defined api_key_env_var={api_key_env_var} but the value is not set. "
                    f"Either remove api_key_env_var or set {api_key_env_var}=<some value>. "
                    "Did you forget to add it to your cluster config?"
                )
        return api_key

    def __del__(self):
        if hasattr(self, "_tunnel") and self._tunnel:
            self._tunnel.stop()

    def _maybe_apply_stop_phrase_removal(
        self, result: dict, remove_stop_phrases: bool, stop_phrases: list[str] | None
    ) -> None:
        if remove_stop_phrases:
            result["generation"] = trim_after_stop_phrases(result["generation"], stop_phrases)

    def _get_tokenizer(self, tokenizer: str | None) -> Union[ServerTokenizer, WrapperAutoTokenizer, None]:
        """Initialize the tokenizer from the string, otherwise initialize the tokenizer endpoint"""
        # Try to initialize the tokenizer from tokenizer string
        for tokenizer_string in [tokenizer, self.model_name_or_path]:
            if tokenizer_string is None:
                continue

            wrapped_tokenizer = self._initialize_tokenizer(tokenizer_string)
            if wrapped_tokenizer is not None:
                return wrapped_tokenizer

        # Try to initialize the tokenizer endpoint
        tokenizer_endpoint = self._get_tokenizer_endpoint()
        if tokenizer_endpoint is not None:
            return tokenizer_endpoint

        # No tokenizer found
        LOG.info(f"No tokenizer found for model: {self.model_name_or_path}")
        return None

    def _get_tokenizer_endpoint(self) -> str | None:
        """Get the tokenizer endpoint if available."""
        return None

    def _initialize_tokenizer(self, tokenizer: str | None) -> WrapperAutoTokenizer | None:
        if tokenizer is None:
            return None
        if isinstance(tokenizer, str):
            return WrapperAutoTokenizer(tokenizer)

    # Public methods for client handlers to call
    def get_supported_params(self) -> set:
        """Public method for models to restrict parameters - override in subclasses"""
        return set()  # Base implementation allows all

    def apply_model_specific_params(self, request: dict, params: dict) -> dict:
        """Public method for model-specific parameter handling - override in subclasses"""
        return request

    def parse_chat_completion_response(self, response, **kwargs) -> dict:
        """Public method for parsing chat completion responses"""
        return self._parse_chat_completion_response(response, **kwargs)

    def parse_completion_response(self, response, **kwargs) -> dict:
        """Public method for parsing completion responses"""
        return self._parse_completion_response(response, **kwargs)

    def parse_responses_response(self, response, **kwargs) -> dict:
        """Public method for parsing responses API responses"""
        result = {"generation": "", "num_generated_tokens": 0}

        # Get token usage - ensure it's always an integer
        if hasattr(response, "usage") and response.usage:
            tokens = getattr(response.usage, "output_tokens", None)
            if tokens is None:
                # Try alternative field names for token usage
                tokens = getattr(response.usage, "completion_tokens", None)
            result["num_generated_tokens"] = tokens if tokens is not None else 0
        else:
            result["num_generated_tokens"] = 0

        # Check for tool calls in the output array
        tool_calls = []
        reasoning_content = ""
        generation_text = ""

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                # Handle reasoning content
                if hasattr(output_item, "type") and output_item.type == "reasoning":
                    if hasattr(output_item, "content") and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, "text"):
                                reasoning_content += content_item.text + "\n"

                # Handle function calls
                elif hasattr(output_item, "type") and output_item.type == "function_call":
                    tool_calls.append(output_item)

                # Handle message content
                elif hasattr(output_item, "type") and output_item.type == "message":
                    if hasattr(output_item, "content") and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, "text"):
                                generation_text += content_item.text

        # Set the appropriate response fields
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["generation"] = ""  # No text generation when there are tool calls
        else:
            result["generation"] = generation_text

        # Add reasoning content if available
        if reasoning_content:
            result["reasoning_content"] = reasoning_content.strip()

        # Add finish reason if available
        if hasattr(response, "status"):
            result["finish_reason"] = response.status

        # Add serialized output for conversation history
        result["serialized_output"] = self._serialize_response_output(response)

        if kwargs.get("include_response", False):
            result["response"] = response

        # Ensure num_generated_tokens is never None for metrics compatibility
        if result["num_generated_tokens"] is None:
            result["num_generated_tokens"] = 0

        return result

    def _serialize_response_output(self, response):
        """Serialize response output objects using model_dump() for conversation history."""
        serialized_output = []

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                try:
                    # Try to use model_dump() method if available (Pydantic models)
                    if hasattr(output_item, "model_dump"):
                        serialized_output.append(output_item.model_dump())
                    # Fallback to dict conversion
                    elif hasattr(output_item, "__dict__"):
                        serialized_output.append(output_item.__dict__)
                    # Last resort: convert to string representation
                    else:
                        serialized_output.append({"content": str(output_item), "type": "unknown"})
                except Exception as e:
                    LOG.warning(f"Failed to serialize output item: {e}")
                    # Fallback serialization
                    serialized_output.append({"content": str(output_item), "type": "error", "error": str(e)})

        return serialized_output

    def _handle_streaming_response(self, response):
        """Handle streaming responses based on response type"""
        if hasattr(response, "choices") and hasattr(response.choices[0], "message"):
            return self._stream_chat_chunks_sync(response)
        else:
            return self._stream_completion_chunks_sync(response)

    @abc.abstractmethod
    def _build_chat_request_params(self, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def _build_completion_request_params(self, **kwargs) -> dict:
        pass

    def _build_request_params(self, prompt: str | list[dict], stream: bool, **kwargs) -> dict:
        if isinstance(prompt, str):
            return self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
        elif isinstance(prompt, list):
            request_params = self._build_chat_request_params(messages=prompt, stream=stream, **kwargs)
            return request_params
        else:
            raise ValueError("Either prompt or messages must be provided")

    @with_context_retry
    async def generate_async(
        self,
        prompt: str | list[dict],
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,  # None is 10min
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """Unified async version of generate for single prompt."""

        # Check tool calls are a list of dict
        if tools is not None:
            for tool in tools:
                # TODO: We may want to add additional checks for tools in the future
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool must be a dictionary, got {type(tool)}")

        # Build kwargs dict explicitly to avoid capturing unwanted local variables
        kwargs = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "random_seed": random_seed,
            "stop_phrases": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "extra_body": extra_body,
            "remove_stop_phrases": remove_stop_phrases,
            "stream": stream,
            "include_response": include_response,
        }

        # TODO: remove this after we no longer use gpt-oss or it's fixed in vllm
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Delegate to client handler using public API
                response = await self.client_handler.call_api_async(prompt, **kwargs)

                if stream:
                    return self._handle_streaming_response(response)
                else:
                    result = self.client_handler.parse_response(response, **kwargs)
                    if remove_stop_phrases:
                        self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
                    return result

            except openai.BadRequestError as e:
                if "output messages (reasoning and final)" in str(e):
                    if retry_count < max_retries:
                        retry_count += 1
                        LOG.warning(f"BadRequestError, retrying {retry_count}/{max_retries}: {e}")
                        continue

                    LOG.error(f"BadRequestError after {max_retries} retries, returning empty response: {e}")
                    return {"generation": "", "reasoning_content": "", "num_generated_tokens": 0}
                else:
                    raise e

        return result

    @with_context_retry
    def generate_sync(
        self,
        prompt: str | list[dict],
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,  # None is 10min
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """
        Unified synchronous version of generate for single prompt.
        See generate_async for full list of parameters.
        """
        # Check tool calls are a list of dict
        if tools is not None:
            for tool in tools:
                # TODO: We may want to add additional checks for tools in the future
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool must be a dictionary, got {type(tool)}")

        # Build kwargs dict explicitly to avoid capturing unwanted local variables
        kwargs = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "random_seed": random_seed,
            "stop_phrases": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "extra_body": extra_body,
            "remove_stop_phrases": remove_stop_phrases,
            "stream": stream,
            "include_response": include_response,
        }

        # Delegate to client handler using public API
        response = self.client_handler.call_api_sync(prompt, **kwargs)

        if stream:
            return self._handle_streaming_response(response)
        else:
            result = self.client_handler.parse_response(response, **kwargs)
            if remove_stop_phrases:
                self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
            return result

    def _parse_completion_response(
        self, response: "openai.types.Completion", include_response: bool = False, **kwargs
    ) -> dict:
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

        result = {"generation": output, "num_generated_tokens": response.usage.completion_tokens}
        if getattr(choice, "logprobs", None):
            result["logprobs"] = choice.logprobs.token_logprobs
            result["tokens"] = choice.logprobs.tokens
            result["top_logprobs"] = choice.logprobs.top_logprobs
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
        result = {"generation": output, "num_generated_tokens": response.usage.completion_tokens}

        # Add reasoning_content if available
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            result["reasoning_content"] = choice.message.reasoning_content

        # Extract detailed token breakdown for reasoning models if available
        if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
            details = response.usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                result["num_reasoning_tokens"] = details.reasoning_tokens
                result["num_answer_tokens"] = response.usage.completion_tokens - details.reasoning_tokens

        if getattr(choice, "logprobs", None) and choice.logprobs.content:
            result["logprobs"] = [tok.logprob for tok in choice.logprobs.content]
            result["tokens"] = [tok.token for tok in choice.logprobs.content]
            result["top_logprobs"] = []
            for token_logprob in choice.logprobs.content:
                logprob = {entry.token: entry.logprob for entry in token_logprob.top_logprobs}
                if token_logprob.token not in logprob:
                    logprob[token_logprob.token] = token_logprob.logprob
                result["top_logprobs"].append(logprob)
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
            # Check for reasoning_content in delta
            reasoning_delta = (
                getattr(chunk.choices[0].delta, "reasoning_content", None)
                if hasattr(chunk.choices[0].delta, "reasoning_content")
                else None
            )
        else:
            cur_delta = chunk.choices[0].text
            reasoning_delta = None

        finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        result = {"generation": cur_delta}

        # Add reasoning_content to result if available
        if reasoning_delta:
            result["reasoning_content"] = reasoning_delta

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
