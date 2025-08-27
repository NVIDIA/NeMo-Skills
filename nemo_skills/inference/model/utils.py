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
import re
from typing import Union

import requests
from transformers import AutoTokenizer

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def trim_after_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


class ServerTokenizer:
    """Class to encode and decode prompts via POST requests to the tokenizer endpoint."""

    def __init__(self, url):
        self.tokenizer_url = url
        self.detokenizer_url = url.replace("/tokenize", "/detokenize")

    def encode(self, prompt: str | list[dict]) -> list[int]:
        """Encode the prompt using the tokenizer endpoint."""
        if isinstance(prompt, str):
            payload = {"prompt": prompt}
        elif isinstance(prompt, list):
            payload = {"messages": prompt}

        response = requests.post(self.tokenizer_url, json=payload, timeout=30)
        response.raise_for_status()

        tokens = response.json()["tokens"]
        return tokens

    def decode(self, tokens: list) -> str:
        """Decode a list of tokens using the tokenizer endpoint."""
        payload = {"tokens": tokens}
        response = requests.post(self.detokenizer_url, json=payload, timeout=30)
        response.raise_for_status()

        text = response.json()["prompt"]
        return text


class WrapperAutoTokenizer:
    """Wrapper around the AutoTokenizer class to provide same interface as the ServerTokenizer class."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, prompt: str | list[dict]) -> list[int]:
        """Encode the prompt using the tokenizer."""
        if isinstance(prompt, str):
            LOG.info(f"Encoding prompt: {prompt[:100]}")
            return self.tokenizer.encode(prompt)
        elif isinstance(prompt, list):
            LOG.info(f"Encoding prompt: {str(prompt)[:1000]}")
            return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens using the tokenizer."""
        return self.tokenizer.decode(tokens)


class RequestException(RuntimeError):
    pass
