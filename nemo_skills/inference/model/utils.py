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

import litellm
import requests

LOG = logging.getLogger(get_logger_name(__file__))


def trim_after_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


def get_tokenizer_endpoint(base_url, model_name):
    """
    Returns the tokenizer endpoint if available, otherwise returns None.
    """
    tokenize_url = base_url.replace("/v1", "/tokenize")
    payload = {"model": model_name, "messages": [{"role": "user", "content": "Test prompt"}]}

    try:
        response = requests.post(tokenize_url, json=payload)
        if response.status_code == 200:
            LOG.info(f"Tokenize endpoint is available! - {tokenize_url}")
            return tokenize_url
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None


def encode(prompt, model, tokenizer_endpoint):
    """
    Encode a prompt using the tokenizer endpoint.
    """
    if isinstance(prompt, str):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    elif isinstance(prompt, list):
        payload = {"model": model, "messages": prompt}
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    try:
        response = requests.post(tokenize_url, json=payload)
        if response.status_code == 200:
            return response.json()['tokens']
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None


class RequestException(RuntimeError):
    pass
