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
from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import MISSING, OmegaConf

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.utils import get_help_message, setup_logging

LOG = logging.getLogger(__file__)


@dataclass
class EvaluateResultsConfig:
    """Top-level parameters for the script"""

    # list of files to evaluate. Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_folder/output-rs*.jsonl"
    prediction_jsonl_files: Any = MISSING
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'local'})
    ignore_cache: bool = False

    include_percentage: bool = True
    tolerance: float = 1e-4

    timeout: float = 10.0
    num_parallel_requests: int = 100
    in_memory_lines: int = 1500

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_evaluate_results_config", node=EvaluateResultsConfig)


@hydra.main(version_base=None, config_name="base_evaluate_results_config")
def evaluate_results(cfg: EvaluateResultsConfig):
    cfg = OmegaConf.to_object(cfg)
    LOG.info("Config used: %s", cfg)

    sandbox = get_sandbox(**cfg.sandbox)
    sandbox.batch_evaluate_results(
        prediction_jsonl_files=cfg.prediction_jsonl_files,
        num_parallel_requests=cfg.num_parallel_requests,
        in_memory_lines=cfg.in_memory_lines,
        include_percentage=cfg.include_percentage,
        tolerance=cfg.tolerance,
        timeout=cfg.timeout,
        ignore_cache=cfg.ignore_cache,
    )


HELP_MESSAGE = get_help_message(
    EvaluateResultsConfig,
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        evaluate_results()
