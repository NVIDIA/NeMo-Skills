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

import glob
import json
import logging
import math
import os
import random
from collections import defaultdict
from copy import deepcopy

import hydra

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def read_file(file_path, output_dir):
    single_answer_instances_path = os.path.join(output_dir, "single_answer_instances.jsonl")

    LOG.info(f"Reading file: {file_path}")
    instances = [json.loads(line) for line in open(file_path, "r")]
    for instance in instances:
        instance["problem"] = instance["question_content"]
        instance["max_idx"] = len(instance["code_list"])

        instance["is_correct"]

        # if "is_correct" not in instance:
        #     if "graded_list" in instance:
        #         instance["is_correct"] = instance["graded_list"][0]
        #     if "judgement" in instance:
        #         instance["is_correct"] = is_correct_judgement(instance["judgement"])

        # if "predicted_answer" not in instance:
        #     if "completion" in instance:
        #         instance["predicted_answer"] = instance["completion"]

    problem_to_instance = {instance["problem"]: instance for instance in instances}
    return problem_to_instance




def preprocess(input_file, output_dir, num_random_seeds=8):
    if output_dir is None:
        raise ValueError("Output directory is required")

    output_dir = os.path.join(output_dir, "comparison_instances")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_instances = read_file(input_file)

    for random_seed in range(num_random_seeds):
        # random.seed(random_seed)
        with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
            for problem, clustered_instances in problem_to_clustered_instances.items():
                comparison_instance = create_comparison_instance(
                    clustered_instances,
                    problem,
                    max_soln_samples=max_soln_samples,
                    sampling_strategy=sampling_strategy,
                )
                f.write(json.dumps(comparison_instance) + "\n")


@nested_dataclass(kw_only=True)
class GenSelectPreprocessConfig:
    input_file: str
    output_dir: str
    num_random_seeds: int | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_preprocess_config", node=GenSelectPreprocessConfig)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_preprocess_config')
def genselect_preprocessor(cfg: GenSelectPreprocessConfig):
    cfg = GenSelectPreprocessConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    preprocess(
        input_file=cfg.input_file,
        output_dir=cfg.output_dir,
        num_random_seeds=cfg.num_random_seeds,
    )


if __name__ == "__main__":
    setup_logging()
    genselect_preprocessor()
