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

"""Simple wrapper that adds a fixed key-value pair to generation results."""

from typing import Any, Dict

from nemo_skills.inference.model.base import BaseModel
from nemo_skills.inference.model.wrapper import ContextAwareModel, ContextAwareWrapper


class AddFixedKeyWrapper(ContextAwareWrapper):
    """Wrapper that adds a fixed key-value pair to results."""

    def default_config(self) -> Dict[str, Any]:
        return {"key": "processed", "value": "true"}

    def wrap(self, model: BaseModel) -> BaseModel:
        return AddFixedKeyModel(model, self.config)


class AddFixedKeyModel(ContextAwareModel):
    """Model that adds a fixed key-value pair to generation results."""

    async def post_process(self, result, data_point, all_data):
        """Add the configured key-value pair to the result."""
        result[self.config["key"]] = self.config["value"]
        return result
