# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import field
from typing import Dict, Tuple

import hydra

from nemo_skills.inference.generate import GenerateSolutionsConfig
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class BaseInspectorConfig:
    model_prediction: Dict[str, str] = field(default_factory=dict)

    code_separators: Tuple[str, str] = ('<llm-code>', '</llm-code>')
    code_output_separators: Tuple[str, str] = ('<llm-code-output>', '</llm-code-output>')
    save_generations_path: str = "nemo_inspector/results/saved_generations"

    def __post_init__(self):
        self.model_prediction = {
            model_name: list(unroll_files(file_path.split(" ")))
            for model_name, file_path in self.model_prediction.items()
        }


@nested_dataclass(kw_only=True)
class InspectorConfig(GenerateSolutionsConfig):
    inspector_params: BaseInspectorConfig = field(default_factory=BaseInspectorConfig)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_inspector_config", node=InspectorConfig)
