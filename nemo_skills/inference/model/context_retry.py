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

from .utils import ServerTokenizer

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
            "max_context_length": int(match.group(1)),
            "total_requested_tokens": int(match.group(2)),
            "message_tokens": int(match.group(3)),
            "completion_tokens": int(match.group(4)),
        }

    return None


@dataclass
class ContextLengthRetry:
    """Configuration for context window retry behavior."""

    enable_soft_fail: bool = False
    strategy: str = "reduce_generation"
    num_special_tokens_budget: int = 10

    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = ["reduce_generation", "reduce_prompt_from_start", "reduce_prompt_from_end"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

        if self.enable_soft_fail:
            LOG.info(f"Soft fail enabled with strategy: {self.strategy}")

    @property
    def reduce_generate_tokens(self):
        """Reduce the number of tokens to generate."""
        LOG.info("Message is too long. Reducing the number of tokens to generate.")
        return self.strategy == "reduce_generation"

    @property
    def reduce_prompt_from_start(self):
        """Remove tokens from the start of the prompt."""
        LOG.info("Message is too long. Removing tokens from the start of the prompt.")
        return self.strategy == "reduce_prompt_from_start"

    @property
    def reduce_prompt_from_end(self):
        """Remove tokens from the end of the prompt."""
        LOG.info("Message is too long. Removing tokens from the end of the prompt.")
        return self.strategy == "reduce_prompt_from_end"


def with_context_retry(func: Callable) -> Callable:
    """
    Decorator to add context window retry logic to generate functions.
    Uses the model's context_retry_config attribute.
    """
    default_config = ContextLengthRetry()

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        config = getattr(self, "context_retry_config", default_config)
        return await handle_context_retries_async(func, self, args, kwargs, config)

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        config = getattr(self, "context_retry_config", default_config)
        return handle_context_retries_sync(func, self, args, kwargs, config)

    # Return the appropriate wrapper based on whether the function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


async def handle_context_retries_async(
    func: Callable, self, args: tuple, kwargs: dict, config: ContextLengthRetry
) -> dict:
    """Sync version of context retry logic."""
    original_tokens_to_generate = kwargs.get("tokens_to_generate", 2048)

    try:
        result = await func(self, *args, **kwargs)
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
                max_context_length = context_error["max_context_length"]
                message_tokens = context_error["message_tokens"]
                completion_tokens = context_error["completion_tokens"]

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
                    kwargs["tokens_to_generate"] = reduced_tokens_to_generate
                    return await func(self, *args, **kwargs)

                # Strategy 2: Reduce the number of tokens in the prompt
                if config.reduce_prompt_from_start or config.reduce_prompt_from_end:
                    # First let's check if token reduction is even feasible for the current config
                    if completion_tokens >= max_context_length:
                        detailed_error = (
                            "Completion tokens are already at the max context length. Cannot reduce prompt tokens.\n\n"
                            + str(e)
                        )
                        LOG.warning(detailed_error)
                        return return_empty_generation_with_error(detailed_error)

                    # Create a copy of args to avoid mutating the original
                    encoded_prompt = self.tokenizer.encode(kwargs["prompt"])
                    LOG.info(f"Length of encoded prompt: {len(encoded_prompt)}")

                    num_prompt_tokens_to_keep = max_context_length - completion_tokens

                    # Reduce the number of tokens in the prompt
                    # If the prompt is a string, we will trim the string
                    # If the prompt is a list, we will remove individual messages from the end or start of the list

                    if isinstance(kwargs["prompt"], str):
                        # Reduce the number of tokens in the prompt
                        if config.reduce_prompt_from_start:
                            trimmed_encoded_prompt = encoded_prompt[-num_prompt_tokens_to_keep:]
                        elif config.reduce_prompt_from_end:
                            trimmed_encoded_prompt = encoded_prompt[:num_prompt_tokens_to_keep]
                        trimmed_prompt = self.tokenizer.decode(trimmed_encoded_prompt)
                        kwargs["prompt"] = trimmed_prompt

                    elif isinstance(kwargs["prompt"], list):
                        # If the prompt is a list, we will remove individual messages from the end or start of the list
                        # Gather length of all prefixes, and then decide where to trim the list

                        # Create a copy of the prompt list to avoid mutating the original
                        prompt_list = copy.deepcopy(kwargs["prompt"])

                        # Iterate over the prompt list and trim the messages
                        prefix_length_list = []
                        trimmed_prompt_messages = []
                        trimmed_prefix_length = 0
                        if config.reduce_prompt_from_end:
                            for idx in range(len(prompt_list)):
                                # Encode messages up to the current message
                                encoded_prefix = self.tokenizer.encode(prompt_list[: idx + 1])
                                if encoded_prefix is None:
                                    continue

                                prefix_length = len(encoded_prefix)
                                if prefix_length > num_prompt_tokens_to_keep:
                                    # Can't add any more messages
                                    # If we have a list of prefixes, we can trim the last message
                                    if prefix_length_list:
                                        # We have a list of prefixes, so we can trim the last message
                                        trimmed_prompt_message_idx, trimmed_prefix_length = prefix_length_list[-1]
                                        trimmed_prompt_messages = prompt_list[:trimmed_prompt_message_idx]

                                    # Trim the current message
                                    # Remaining tokens for the current message
                                    num_rem_tokens = num_prompt_tokens_to_keep - trimmed_prefix_length
                                    cur_trimmed_content = get_trimmed_content(
                                        content=prompt_list[idx]["content"],
                                        num_rem_tokens=num_rem_tokens,
                                        num_special_tokens_budget=config.num_special_tokens_budget,
                                        tokenizer=self.tokenizer,
                                        trim_suffix=True,  # Trim the suffix of the message
                                    )
                                    if cur_trimmed_content is not None:
                                        # We can add the current message by trimming its content
                                        prompt_list[idx]["content"] = cur_trimmed_content
                                        trimmed_prompt_messages.append(prompt_list[idx])

                                    break

                                else:
                                    # Add details to the prefix length list
                                    # We are adding the index of the message corresponding to the prefix length
                                    # This index is necessary to avoid errors in case the subsequent prefix_lengths are None
                                    prefix_length_list.append((idx + 1, prefix_length))

                        elif config.reduce_prompt_from_start:
                            for idx in range(len(prompt_list)):
                                # Encode messages up to the current message
                                encoded_prefix = self.tokenizer.encode(prompt_list[-(idx + 1) :])

                                if encoded_prefix is None:
                                    continue

                                prefix_length = len(encoded_prefix)
                                if prefix_length > num_prompt_tokens_to_keep:
                                    # Can't add any more messages
                                    if prefix_length_list:
                                        # We have a list of prefixes, so we can trim the last message
                                        trimmed_prompt_message_idx, trimmed_prefix_length = prefix_length_list[-1]
                                        trimmed_prompt_messages = prompt_list[trimmed_prompt_message_idx:]

                                    # Trim the current message
                                    # Remaining tokens for the current message
                                    num_rem_tokens = num_prompt_tokens_to_keep - trimmed_prefix_length
                                    cur_trimmed_content = get_trimmed_content(
                                        content=prompt_list[-(idx + 1)]["content"],
                                        num_rem_tokens=num_rem_tokens,
                                        num_special_tokens_budget=config.num_special_tokens_budget,
                                        tokenizer=self.tokenizer,
                                        trim_suffix=False,  # Trim the suffix of the message
                                    )
                                    if cur_trimmed_content is not None:
                                        # We can add the current message by trimming its content
                                        prompt_list[-(idx + 1)]["content"] = cur_trimmed_content
                                        trimmed_prompt_messages.insert(0, prompt_list[-(idx + 1)])

                                    break

                                else:
                                    # Add details to the prefix length list
                                    # We are adding the index of the message corresponding to the prefix length
                                    # This index is necessary to avoid errors in case the subsequent prefix_lengths are None
                                    prefix_length_list.append((-(idx + 1), prefix_length))

                        if trimmed_prompt_messages:
                            # Set the prompt to be the reduced list of messages
                            kwargs["prompt"] = trimmed_prompt_messages
                        else:
                            detailed_error = "Not able to trim the prompt. Returning empty generation.\n\n" + str(e)
                            LOG.warning(detailed_error)
                            return return_empty_generation_with_error(detailed_error)

                    return await func(self, *args, **kwargs)


def handle_context_retries_sync(func: Callable, self, args: tuple, kwargs: dict, config: ContextLengthRetry) -> dict:
    """Sync version of context retry logic."""
    original_tokens_to_generate = kwargs.get("tokens_to_generate", 2048)

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
                max_context_length = context_error["max_context_length"]
                message_tokens = context_error["message_tokens"]
                completion_tokens = context_error["completion_tokens"]

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
                    kwargs["tokens_to_generate"] = reduced_tokens_to_generate
                    return func(self, *args, **kwargs)

                # Strategy 2: Reduce the number of tokens in the prompt
                if config.reduce_prompt_from_start or config.reduce_prompt_from_end:
                    # First let's check if token reduction is even feasible for the current config
                    if completion_tokens >= max_context_length:
                        detailed_error = (
                            "Completion tokens are already at the max context length. Cannot reduce prompt tokens.\n\n"
                            + str(e)
                        )
                        LOG.warning(detailed_error)
                        return return_empty_generation_with_error(detailed_error)

                    # Create a copy of args to avoid mutating the original
                    encoded_prompt = self.tokenizer.encode(kwargs["prompt"])
                    LOG.info(f"Length of encoded prompt: {len(encoded_prompt)}")

                    num_prompt_tokens_to_keep = max_context_length - completion_tokens

                    # Reduce the number of tokens in the prompt
                    # If the prompt is a string, we will trim the string
                    # If the prompt is a list, we will remove individual messages from the end or start of the list

                    if isinstance(kwargs["prompt"], str):
                        # Reduce the number of tokens in the prompt
                        if config.reduce_prompt_from_start:
                            trimmed_encoded_prompt = encoded_prompt[-num_prompt_tokens_to_keep:]
                        elif config.reduce_prompt_from_end:
                            trimmed_encoded_prompt = encoded_prompt[:num_prompt_tokens_to_keep]
                        trimmed_prompt = self.tokenizer.decode(trimmed_encoded_prompt)
                        kwargs["prompt"] = trimmed_prompt

                    elif isinstance(kwargs["prompt"], list):
                        # If the prompt is a list, we will remove individual messages from the end or start of the list
                        # Gather length of all prefixes, and then decide where to trim the list

                        # Create a copy of the prompt list to avoid mutating the original
                        prompt_list = copy.deepcopy(kwargs["prompt"])

                        # Iterate over the prompt list and trim the messages
                        prefix_length_list = []
                        trimmed_prompt_messages = []
                        trimmed_prefix_length = 0
                        if config.reduce_prompt_from_end:
                            for idx in range(len(prompt_list)):
                                # Encode messages up to the current message
                                encoded_prefix = self.tokenizer.encode(prompt_list[: idx + 1])
                                if encoded_prefix is None:
                                    continue

                                prefix_length = len(encoded_prefix)
                                if prefix_length > num_prompt_tokens_to_keep:
                                    # Can't add any more messages
                                    # If we have a list of prefixes, we can trim the last message
                                    if prefix_length_list:
                                        # We have a list of prefixes, so we can trim the last message
                                        trimmed_prompt_message_idx, trimmed_prefix_length = prefix_length_list[-1]
                                        trimmed_prompt_messages = prompt_list[:trimmed_prompt_message_idx]

                                    # Trim the current message
                                    # Remaining tokens for the current message
                                    num_rem_tokens = num_prompt_tokens_to_keep - trimmed_prefix_length
                                    cur_trimmed_content = get_trimmed_content(
                                        content=prompt_list[idx]["content"],
                                        num_rem_tokens=num_rem_tokens,
                                        num_special_tokens_budget=config.num_special_tokens_budget,
                                        tokenizer=self.tokenizer,
                                        trim_suffix=True,  # Trim the suffix of the message
                                    )
                                    if cur_trimmed_content is not None:
                                        # We can add the current message by trimming its content
                                        prompt_list[idx]["content"] = cur_trimmed_content
                                        trimmed_prompt_messages.append(prompt_list[idx])

                                    break

                                else:
                                    # Add details to the prefix length list
                                    # We are adding the index of the message corresponding to the prefix length
                                    # This index is necessary to avoid errors in case the subsequent prefix_lengths are None
                                    prefix_length_list.append((idx + 1, prefix_length))

                        elif config.reduce_prompt_from_start:
                            for idx in range(len(prompt_list)):
                                # Encode messages up to the current message
                                encoded_prefix = self.tokenizer.encode(prompt_list[-(idx + 1) :])

                                if encoded_prefix is None:
                                    continue

                                prefix_length = len(encoded_prefix)
                                if prefix_length > num_prompt_tokens_to_keep:
                                    # Can't add any more messages
                                    if prefix_length_list:
                                        # We have a list of prefixes, so we can trim the last message
                                        trimmed_prompt_message_idx, trimmed_prefix_length = prefix_length_list[-1]
                                        trimmed_prompt_messages = prompt_list[trimmed_prompt_message_idx:]

                                    # Trim the current message
                                    # Remaining tokens for the current message
                                    num_rem_tokens = num_prompt_tokens_to_keep - trimmed_prefix_length
                                    cur_trimmed_content = get_trimmed_content(
                                        content=prompt_list[-(idx + 1)]["content"],
                                        num_rem_tokens=num_rem_tokens,
                                        num_special_tokens_budget=config.num_special_tokens_budget,
                                        tokenizer=self.tokenizer,
                                        trim_suffix=False,  # Trim the suffix of the message
                                    )
                                    if cur_trimmed_content is not None:
                                        # We can add the current message by trimming its content
                                        prompt_list[-(idx + 1)]["content"] = cur_trimmed_content
                                        trimmed_prompt_messages.insert(0, prompt_list[-(idx + 1)])

                                    break

                                else:
                                    # Add details to the prefix length list
                                    # We are adding the index of the message corresponding to the prefix length
                                    # This index is necessary to avoid errors in case the subsequent prefix_lengths are None
                                    prefix_length_list.append((-(idx + 1), prefix_length))

                        if trimmed_prompt_messages:
                            # Set the prompt to be the reduced list of messages
                            kwargs["prompt"] = trimmed_prompt_messages
                        else:
                            detailed_error = "Not able to trim the prompt. Returning empty generation.\n\n" + str(e)
                            LOG.warning(detailed_error)
                            return return_empty_generation_with_error(detailed_error)

                    return func(self, *args, **kwargs)


def get_trimmed_content(
    content: str,
    num_rem_tokens: int,
    num_special_tokens_budget: int,
    tokenizer: ServerTokenizer,
    trim_suffix: bool = True,
) -> str:
    """
    Get the trimmed content of a message.
    """
    # Remove the budget for special tokens
    if num_rem_tokens > num_special_tokens_budget:
        num_rem_tokens = num_rem_tokens - num_special_tokens_budget
        encoded_content = tokenizer.encode(content)
        if trim_suffix:
            encoded_content = encoded_content[:num_rem_tokens]
        else:
            encoded_content = encoded_content[-num_rem_tokens:]
        trimmed_content = tokenizer.decode(encoded_content)
        return trimmed_content
    else:
        return None


def return_empty_generation_with_error(detailed_error: str):
    return {
        "generation": "",
        "num_generated_tokens": 0,
        "error": "context_window_exceeded",
        "detailed_error": detailed_error,
    }
