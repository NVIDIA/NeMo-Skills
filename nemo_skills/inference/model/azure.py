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

import os
import threading

from openai import AzureOpenAI

from .base import BaseModel
from .openai import OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        model=None,
        base_url=None,
        api_key=None,
        api_version: str = "2024-02-15-preview",
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        **kwargs,
    ):
        # TODO: fix that by moving non-relevant logic into methods and overriding
        # Call BaseModel.__init__ directly to bypass OpenAIModel.__init__ logic
        BaseModel.__init__(self, host=host, port=port, **kwargs)
        self._tunnel = None

        # Track active generations with thread-safe operations
        self.active_generations = {}
        self._generations_lock = threading.Lock()

        model = model or os.getenv("NEMO_SKILLS_OPENAI_MODEL")
        if model is None:
            raise ValueError("model argument is required for Azure OpenAI model.")

        azure_endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError("base_url/AZURE_OPENAI_ENDPOINT is required for Azure OpenAI model.")

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("api_key/AZURE_OPENAI_API_KEY is required for Azure OpenAI model.")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
