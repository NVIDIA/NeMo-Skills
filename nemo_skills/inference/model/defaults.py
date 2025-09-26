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

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationDefaults:
    """Shared defaults for generation parameters across all models and handlers"""

    tokens_to_generate: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    random_seed: Optional[int] = None
    stop_phrases: Optional[List[str]] = None
    top_logprobs: Optional[int] = None
    timeout: Optional[float] = 14400
    remove_stop_phrases: bool = True
    stream: bool = False
    reasoning_effort: Optional[str] = None
    tools: Optional[List[dict]] = None
    include_response: bool = False
    extra_body: Optional[dict] = None


# Base parameter sets for different client types
CHAT_COMPLETION_PARAMS = {
    "tokens_to_generate",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "random_seed",
    "stop_phrases",
    "top_logprobs",
    "timeout",
    "stream",
    "reasoning_effort",
    "tools",
    "extra_body",
    "remove_stop_phrases",
    "include_response",
}

RESPONSES_PARAMS = {
    "tokens_to_generate",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "random_seed",
    "stop_phrases",
    "top_logprobs",
    "timeout",
    "stream",
    "reasoning_effort",
    "tools",
    "extra_body",
    "remove_stop_phrases",
    "include_response",
}

COMPLETION_PARAMS = {
    "tokens_to_generate",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "random_seed",
    "stop_phrases",
    "top_logprobs",
    "timeout",
    "stream",
    "extra_body",
    "remove_stop_phrases",
    "include_response",
    # Note: no 'reasoning_effort' or 'tools' for completion
}
