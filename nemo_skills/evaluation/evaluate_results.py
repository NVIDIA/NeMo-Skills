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

import json
import logging
import sys
from dataclasses import field
from typing import Any

import hydra

from nemo_skills.evaluation.evaluator import evaluate
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    remove_thinking,
    setup_logging,
    unroll_files,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class EvaluateResultsConfig:
    """Top-level parameters for the script"""

    # list of files to evaluate. Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_dir/output-rs*.jsonl"
    input_files: Any

    eval_type: str
    # the supported parameters are different depending on the eval configuration
    # check graders.py for the supported eval types and their parameters
    eval_config: dict = field(default_factory=dict)

    # whether to remove the thinking part from the final output
    remove_thinking: bool = True
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"
    # generation key in the jsonl file
    generation_key: str = "generation"

    def __post_init__(self):
        if isinstance(self.input_files, str):
            self.input_files = self.input_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_evaluate_results_config", node=EvaluateResultsConfig)


@hydra.main(version_base=None, config_name="base_evaluate_results_config")
def evaluate_results(cfg: EvaluateResultsConfig):
    cfg = EvaluateResultsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    if cfg.remove_thinking:
        LOG.info(
            f"Removing the thinking part from the {cfg.generation_key} key "
            f"(using {cfg.thinking_begin} and {cfg.thinking_end} tokens). "
            'Original content will be stored in "_full_generation" key.'
        )
        for jsonl_file in unroll_files(cfg.input_files):
            with open(jsonl_file, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f]
                for sample in samples:
                    if cfg.generation_key not in sample:
                        raise ValueError(
                            f"Key {cfg.generation_key} not found in a sample, but remove_thinking=True is specified. "
                            "Use generation_key parameter to specify the key containing the generations."
                        )
            with open(jsonl_file, "wt", encoding="utf-8") as f:
                for sample in samples:
                    remove_thinking(sample, cfg.generation_key, cfg.thinking_begin, cfg.thinking_end)
                    f.write(json.dumps(sample) + "\n")

    evaluate(cfg)


HELP_MESSAGE = get_help_message(
    EvaluateResultsConfig,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        evaluate_results()
