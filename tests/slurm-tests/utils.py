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

import json
from pathlib import Path


def load_json(path: Path | str):
    """Load a JSON file from the given path."""
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(nested_dict: dict, nested_keys: tuple | list):
    for k in nested_keys:
        if not isinstance(nested_dict, dict) or k not in nested_dict:
            return None
        nested_dict = nested_dict[k]
    # resolves to the value eventually
    return nested_dict
