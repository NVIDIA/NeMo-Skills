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

import logging
import sys
from dataclasses import field
from typing import Any

import hydra
from omegaconf import MISSING

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.evaluation.settings import GRADING_MAP
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass
class EvaluateResultsConfig:
    """Top-level parameters for the script"""

    # list of files to evaluate. Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_folder/output-rs*.jsonl"
    prediction_jsonl_files: Any = MISSING

    eval_type: str = "math"
    # the supported parameters are different depending on the eval configuration
    # check graders.py for the supported eval types and their parameters
    eval_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_evaluate_results_config", node=EvaluateResultsConfig)


@hydra.main(version_base=None, config_name="base_evaluate_results_config")
def evaluate_results(cfg: EvaluateResultsConfig):
    cfg = EvaluateResultsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    if cfg.eval_type not in GRADING_MAP:
        raise ValueError(f"Unknown eval_type: {cfg.eval_type}")

    GRADING_MAP[cfg.eval_type](cfg)


HELP_MESSAGE = get_help_message(
    EvaluateResultsConfig,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        evaluate_results()
