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
import shutil
import sys
from argparse import Namespace
from dataclasses import field
from typing import Any

import hydra
from omegaconf import MISSING, OmegaConf

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.evaluation.code_utils import preprocess_code
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass
class EvaluateResultsConfig:
    """Top-level parameters for the script"""

    # list of files to evaluate. Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_folder/output-rs*.jsonl"
    prediction_jsonl_files: Any = MISSING
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'local'})

    eval_type: str = "math"  # math or code
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

    sandbox = get_sandbox(**cfg.sandbox)
    if cfg.eval_type == "math":
        sandbox.batch_evaluate_results(
            prediction_jsonl_files=cfg.prediction_jsonl_files,
            **cfg.eval_config,
        )
    elif cfg.eval_type == "code":  # using evalplus for code directly
        # TODO: need to move it to a separate docker (either our sandbox or separate srun)
        from evalplus.evaluate import evaluate

        # processing each generation separately (TODO: evalplus can do it together, but need to figure out the format)
        for jsonl_file in cfg.prediction_jsonl_files:
            with open(jsonl_file) as f:
                samples = [preprocess_code(json.loads(line)) for line in f]
            # all changes will be done with a new key "completion", so it's ok to write to the same file
            with open(jsonl_file, "wt", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            eval_config = {
                "samples": jsonl_file,
                "base_only": False,
                "parallel": None,
                "i_just_wanna_run": False,
                "test_details": False,
                "min_time_limit": 1,
                "gt_time_limit_factor": 4.0,
                "mini": False,
                "noextreme": False,
                "version": "default",
            }
            eval_config.update(OmegaConf.to_container(cfg.eval_config))
            evaluate(Namespace(**eval_config))
            with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
                evalplus_grades = json.load(fin)
            # adding is_correct key to allow compute_metrics to work
            with open(jsonl_file, "wt", encoding="utf-8") as f:
                for sample in samples:
                    sample['is_correct'] = evalplus_grades['eval'][sample['task_id']][0]['base_status'] == "pass"
                    sample['is_correct-plus'] = (
                        sample['is_correct'] and evalplus_grades['eval'][sample['task_id']][0]['plus_status'] == "pass"
                    )
                    f.write(json.dumps(sample) + "\n")

            # moving eval file as otherwise evalplus does not want to recompute metrics if it's present..
            shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')


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
