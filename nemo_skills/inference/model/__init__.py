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

import dataclasses

from nemo_skills.utils import python_doc_to_cmd_help

from .azure import AzureOpenAIModel

# Base classes
from .base import BaseModel

# Code execution
from .code_execution import CodeExecutionConfig, CodeExecutionWrapper
from .context_retry import ContextLimitRetryConfig
from .gemini import GeminiModel
from .megatron import MegatronModel
from .openai import OpenAIModel
from .parallel_thinking import ParallelThinkingConfig, ParallelThinkingTask

# Tool Calling
from .tool_call import ToolCallingWrapper

# Utilities
from .vllm import VLLMModel

# Model implementations


# Model registry
models = {
    "trtllm": VLLMModel,
    "megatron": MegatronModel,
    "openai": OpenAIModel,
    "azureopenai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "vllm": VLLMModel,
    "sglang": VLLMModel,
}


def get_model(server_type, use_responses_api=False, tokenizer=None, **kwargs):
    """A helper function to make it easier to set server through cmd.

    Args:
        server_type: Type of server to use (e.g., 'openai', 'vllm', etc.)
        use_responses_api: Whether to use responses API instead of chat completion API
        tokenizer: Tokenizer to use
        **kwargs: Additional arguments to pass to the model
    """
    model_class = models[server_type.lower()]
    if server_type == "trtllm" and kwargs.get("enable_soft_fail", False):
        if kwargs.get("context_limit_retry_strategy", None) is not None:
            raise ValueError("context_limit_retry_strategy is not supported for trtllm")
    return model_class(use_responses_api=use_responses_api, tokenizer=tokenizer, **kwargs)


def get_code_execution_model(
    server_type, use_responses_api=False, tokenizer=None, code_execution=None, sandbox=None, **kwargs
):
    """A helper function to make it easier to set server through cmd.

    Args:
        server_type: Type of server to use (e.g., 'openai', 'vllm', etc.)
        use_responses_api: Whether to use responses API instead of chat completion API
        tokenizer: Tokenizer to use
        code_execution: Code execution configuration
        sandbox: Sandbox to use for code execution
        **kwargs: Additional arguments to pass to the model
    """
    model = get_model(server_type=server_type, use_responses_api=use_responses_api, tokenizer=tokenizer, **kwargs)
    if code_execution is None:
        code_execution = {}
    code_execution_config = CodeExecutionConfig(**code_execution)
    return CodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)


def get_parallel_thinking_model(
    model,
    orig_prompt_filler,
    parallel_thinking: ParallelThinkingConfig = None,
    main_config=None,
    inference_override_config=None,
):
    """A helper function to create the Parallel Thinking model."""
    # Merging priority: parallel_thinking_config, main config, any overrides from inference_override_config
    merged_config = {
        **parallel_thinking.__dict__,
        **main_config.__dict__,
        **(inference_override_config if inference_override_config is not None else {}),
    }

    # Filter to only include valid parameters
    valid_params = {field.name for field in dataclasses.fields(ParallelThinkingConfig)}
    filtered_config = {key: value for key, value in merged_config.items() if key in valid_params}

    parallel_thinking_config = ParallelThinkingConfig(**filtered_config)

    return ParallelThinkingTask(model=model, orig_prompt_filler=orig_prompt_filler, cfg=parallel_thinking_config)


def get_tool_calling_model(
    model,
    use_responses_api=False,
    tokenizer=None,
    additional_config=None,
    tool_modules: list[str] | None = None,
    tool_overrides: dict | None = None,
    **kwargs,
):
    """A helper function to create a tool calling model.

    Args:
        model: Model name/path or model instance
        use_responses_api: Whether to use responses API instead of chat completion API
        tokenizer: Tokenizer to use
        additional_config: Additional configuration
        tool_modules: List of tool modules to use
        tool_overrides: Tool overrides
        **kwargs: Additional arguments to pass to the model
    """
    if isinstance(model, str):
        model = get_model(model=model, use_responses_api=use_responses_api, tokenizer=tokenizer, **kwargs)
    return ToolCallingWrapper(
        model,
        tool_modules=tool_modules,
        tool_overrides=tool_overrides,
        additional_config=additional_config,
    )


def server_params():
    """Returns server documentation (to include in cmd help)."""
    # TODO: This needs a fix now
    prefix = f"\n        server_type: str = MISSING - Choices: {list(models.keys())}"
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")
