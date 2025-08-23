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

import asyncio
import copy
import functools
import logging
import re
from dataclasses import asdict, dataclass
from typing import Callable, Union

import litellm

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def parse_context_window_exceeded_error(error: litellm.ContextWindowExceededError) -> Union[dict, None]:
    """
    Extract token information from LiteLLM context window error messages.

    Returns:
        Dict with keys: max_context_length, total_requested_tokens, message_tokens, completion_tokens
        None if parsing fails
    """
    # Handle both patterns: with and without parentheses
    pattern1 = re.compile(
        r"maximum context length is (\d+) tokens.*?"
        r"you requested (\d+) tokens.*?"
        r"\((\d+) in the messages, (\d+) in the completion\)",
        re.IGNORECASE | re.DOTALL,
    )

    # Alternative pattern for messages like: "45008 in the messages, 2048 in the completion"
    pattern2 = re.compile(
        r"maximum context length is (\d+) tokens.*?"
        r"you requested (\d+) tokens.*?"
        r"(\d+) in the messages, (\d+) in the completion",
        re.IGNORECASE | re.DOTALL,
    )

    error_str = str(error)

    match = pattern1.search(error_str)
    if not match:
        match = pattern2.search(error_str)

    if match:
        return {
            'max_context_length': int(match.group(1)),
            'total_requested_tokens': int(match.group(2)),
            'message_tokens': int(match.group(3)),
            'completion_tokens': int(match.group(4)),
        }

    return None


@dataclass
class ContextLengthRetry:
    """Configuration for context window retry behavior."""

    enable_soft_fail: bool = False
    # Strategy choices - "reduce_generation", "reduce_prompt_start", "reduce_prompt_end"
    strategy: str = "reduce_generation"
    prompt_tokens_reduction_factor: float = 0.95

    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = ["reduce_generation", "reduce_prompt_start", "reduce_prompt_end"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

        if self.prompt_tokens_reduction_factor > 1 or self.prompt_tokens_reduction_factor < 0:
            raise ValueError("prompt_tokens_reduction_factor must be between 0 and 1")

        if self.enable_soft_fail:
            LOG.info(f"Soft fail enabled with strategy: {self.strategy}")

    @property
    def reduce_generate_tokens(self):
        return self.strategy == "reduce_generation"

    @property
    def reduce_prompt_tokens_start(self):
        return self.strategy == "reduce_prompt_start"

    @property
    def reduce_prompt_tokens_end(self):
        return self.strategy == "reduce_prompt_end"


def with_context_retry(func: Callable) -> Callable:
    """
    Decorator to add context window retry logic to generate functions.
    Uses the model's context_retry_config attribute.
    """

    # @functools.wraps(func)
    # async def async_wrapper(self, *args, **kwargs):
    #     config = getattr(self, 'context_retry_config', ContextLengthRetry())
    #     return await _handle_context_retries_async(func, self, args, kwargs, config)

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        config = getattr(self, 'context_retry_config', ContextLengthRetry())
        return _handle_context_retries_sync(func, self, args, kwargs, config)

    # Return the appropriate wrapper based on whether the function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# async def _handle_context_retries_async(
#     func: Callable, self, args: tuple, kwargs: dict, config: ContextLengthRetry
# ) -> dict:
#     """Async version of context retry logic - mirrors sync behavior."""
#     original_tokens_to_generate = kwargs.get('tokens_to_generate', 2048)

#     try:
#         result = await func(self, *args, **kwargs)
#         return result

#     except litellm.ContextWindowExceededError as e:
#         if not config.enable_soft_fail:
#             raise e
#         else:
#             # Soft fail is enabled. We will try to reduce the number of requested tokens.
#             context_error = parse_context_window_exceeded_error(e)
#             if context_error is None:
#                 # Not able to parse the result. We will just return an empty generation
#                 LOG.warning("Not able to parse the context window exceeded error. Returning empty generation.")
#                 return {
#                     "generation": "",
#                     "num_generated_tokens": 0,
#                     "error": "context_window_exceeded",
#                     "detailed_error": str(e),
#                 }
#             else:
#                 # We were able to parse the result. We will try to reduce the number of requested tokens.
#                 max_context_length = context_error['max_context_length']
#                 message_tokens = context_error['message_tokens']
#                 completion_tokens = context_error['completion_tokens']

#                 # Strategy 1: Reduce the number of tokens to generate
#                 if config.reduce_generate_tokens:
#                     # First let's check if token reduction is even feasible for the current config
#                     if message_tokens >= max_context_length:
#                         LOG.warning(
#                             "Messages tokens are already at the max context length. " "Cannot reduce generate tokens."
#                         )
#                         return {
#                             "generation": "",
#                             "num_generated_tokens": 0,
#                             "error": "context_window_exceeded",
#                             "detailed_error": "prompt tokens already exceed the max context length\n" + str(e),
#                         }

#                     # We can reduce the number of tokens to generate
#                     reduced_generation_budget = max_context_length - message_tokens
#                     # This min operation is probably not needed but just in case
#                     reduced_tokens_to_generate = min(original_tokens_to_generate, reduced_generation_budget)
#                     LOG.warning(
#                         f"Reducing tokens_to_generate from {original_tokens_to_generate} to {reduced_tokens_to_generate} to stay within the context window."
#                     )
#                     kwargs['tokens_to_generate'] = reduced_tokens_to_generate
#                     return await func(self, *args, **kwargs)

#                 # Strategy 2: Reduce the number of tokens in the prompt
#                 if config.reduce_prompt_tokens_start or config.reduce_prompt_tokens_end:
#                     # First let's check if token reduction is even feasible for the current config
#                     if completion_tokens >= max_context_length:
#                         LOG.warning(
#                             "Completion tokens are already at the max context length. " "Cannot reduce prompt tokens."
#                         )
#                         return {
#                             "generation": "",
#                             "num_generated_tokens": 0,
#                             "error": "context_window_exceeded",
#                             "detailed_error": "tokens_to_generate exceed the max context length\n" + str(e),
#                         }

#                     num_prompt_tokens_to_keep = max_context_length - completion_tokens
#                     # Calculate tokens using the tokenizer endpoint
#                     if self.tokenizer_endpoint is not None:
#                         # tokenizer_endpoint = get_tokenizer_endpoint(self.base_url, self.model_name)
#                         # prompt_tokens_to_keep = self.tokenizer_endpoint.encode(kwargs['prompt'], add_special_tokens=True)
#                         pass

#                     return await func(self, *tuple(args), **kwargs)


def _handle_context_retries_sync(func: Callable, self, args: tuple, kwargs: dict, config: ContextLengthRetry) -> dict:
    """Sync version of context retry logic."""
    original_tokens_to_generate = kwargs.get('tokens_to_generate', 2048)

    try:
        result = func(self, *args, **kwargs)
        return result

    except litellm.exceptions.ContextWindowExceededError as e:
        if not config.enable_soft_fail:
            raise e
        else:
            # Soft fail is enabled. We will try to reduce the number of requested tokens.
            context_error = parse_context_window_exceeded_error(e)
            if context_error is None:
                # Not able to parse the result. We will just return an empty generation
                detailed_error = (
                    "Not able to parse the context window exceeded error. Returning empty generation.\n\n" + str(e)
                )
                LOG.warning(detailed_error)
                return return_empty_generation_with_error(str(e))
            else:
                # We were able to parse the result. We will try to reduce the number of requested tokens.
                max_context_length = context_error['max_context_length']
                message_tokens = context_error['message_tokens']
                completion_tokens = context_error['completion_tokens']

                # Strategy 1: Reduce the number of tokens to generate
                if config.reduce_generate_tokens:
                    # First let's check if token reduction is even feasible for the current config
                    if message_tokens >= max_context_length:
                        detailed_error = (
                            "Messages tokens are already at the max context length. Cannot reduce generate tokens.\n\n"
                            + str(e)
                        )
                        LOG.warning(detailed_error)
                        return return_empty_generation_with_error(detailed_error)

                    # We can reduce the number of tokens to generate
                    reduced_generation_budget = max_context_length - message_tokens
                    # This min operation is probably not needed but just in case
                    reduced_tokens_to_generate = min(original_tokens_to_generate, reduced_generation_budget)
                    LOG.warning(
                        f"Reducing tokens_to_generate from {original_tokens_to_generate} to {reduced_tokens_to_generate} to stay within the context window."
                    )
                    kwargs['tokens_to_generate'] = reduced_tokens_to_generate
                    return func(self, *args, **kwargs)

                # Strategy 2: Reduce the number of tokens in the prompt
                if config.reduce_prompt_tokens_start or config.reduce_prompt_tokens_end:
                    # First let's check if token reduction is even feasible for the current config
                    if completion_tokens >= max_context_length:
                        detailed_error = (
                            "Completion tokens are already at the max context length. Cannot reduce prompt tokens.\n\n"
                            + str(e)
                        )
                        LOG.warning(detailed_error)
                        return return_empty_generation_with_error(detailed_error)

                    # Create a copy of args to avoid mutating the original
                    prompt = copy.deepcopy(args[0])
                    encoded_prompt = self.tokenizer_endpoint.encode(prompt)
                    LOG.info(f"Length of encoded prompt: {len(encoded_prompt)}")

                    num_prompt_tokens_to_keep = max_context_length - completion_tokens

                    # Reduce the number of tokens in the prompt
                    # If the prompt is a string, we will trim the string
                    # If the prompt is a list, we will remove individual messages from the end or start of the list
                    if isinstance(prompt, str):
                        # Reduce the number of tokens in the prompt
                        if config.reduce_prompt_tokens_start:
                            trimmed_encoded_prompt = encoded_prompt[-num_prompt_tokens_to_keep:]
                        elif config.reduce_prompt_tokens_end:
                            trimmed_encoded_prompt = encoded_prompt[:num_prompt_tokens_to_keep]
                        trimmed_prompt = self.tokenizer_endpoint.decode(trimmed_encoded_prompt)
                        args[0] = trimmed_prompt

                    elif isinstance(prompt, list):
                        # If the prompt is a list, we will remove individual messages from the end or start of the list
                        # Gather length of all prefixes, and then decide where to trim the list
                        prefix_length_list = []
                        trimmed_prompt_messages = None
                        if config.reduce_prompt_tokens_end:
                            for idx in range(len(prompt)):
                                prefix_length = self.tokenizer_endpoint.encode(prompt[:idx])
                                if prefix_length is not None:
                                    if prefix_length > num_prompt_tokens_to_keep:
                                        # Can't add any more messages
                                        if prefix_length_list:
                                            # We have a list of prefixes, so we can trim the last message
                                            trimmed_prompt_messages = prefix_length_list[-1]["prompt"]
                                        break

                                    prefix_length_list.append({"prompt": prompt[:idx], "length": prefix_length})
                        elif config.reduce_prompt_tokens_start:
                            for num_last_messages in range(len(prompt)):
                                pass

                        if trimmed_prompt_messages is not None:
                            args[0] = trimmed_prompt_messages
                        else:
                            detailed_error = "Not able to trim the prompt. Returning empty generation.\n\n" + str(e)
                            LOG.warning(detailed_error)
                            return return_empty_generation_with_error(detailed_error)

                    return func(self, *args, **kwargs)


def return_empty_generation_with_error(detailed_error: str):
    return {
        "generation": "",
        "num_generated_tokens": 0,
        "error": "context_window_exceeded",
        "detailed_error": detailed_error,
    }
