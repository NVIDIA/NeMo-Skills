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
import random
import re
import sys
from copy import deepcopy
from dataclasses import field
from enum import Enum
from os import makedirs, path
from pathlib import Path
from typing import Any

import hydra
import typer
from tqdm import tqdm

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class GenSelectConfig(GenerateSolutionsConfig):
    """Genselect parameters."""

    input_file: str
    output_file: str

    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    prompt_config: str = "openmath/genselect"  # GenSelect template

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    generation_key: str = "genselect_comparison"
    input_key: str = "problem"
    output_key: str = "generation"
    answer_key: str
    benchmark: str

    sandbox: dict = field(default_factory=dict)

    def _post_init_validate_data(self):
        super()._post_init_validate_data()
        if self.inference.random_seed is None:
            raise ValueError("Random seed is required for genselect")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_config", node=GenSelectConfig)


class GenSelectTask(GenerationTask):
    def __init__(self, cfg: GenSelectConfig):
        super().__init__(cfg)

    def get_generation_module(self):
        return "nemo_skills.inference.genselect"

    def _extract_judgment(self, generation, max_idx=None):
        """Extract the judgment from the generation."""
        judgment = None

        try:
            matches = re.findall(r"Judg[e]?ment: (\d+)", generation)
            # print(matches)

            if matches:
                number = matches[-1]
                judgment = int(number)
                if max_idx is not None and judgment > max_idx:
                    judgment = None
            else:
                judgment = None

        except:
            judgment = None

        if judgment is not None and max_idx is not None:
            if judgment > max_idx:
                judgment = None

        return judgment

    def postprocess(self):
        print("Postprocessing...")
        single_correctness_instances_file = Path(self.cfg.input_file).parent / "single_correctness_instances.jsonl"
        single_correctness_instances = [json.loads(line) for line in open(single_correctness_instances_file, "r")]

        input_file = self.cfg.output_file
        output_file = Path(self.cfg.output_file).parent.parent / self.cfg.benchmark / Path(self.cfg.output_file).name
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, 'r') as f, open(output_file, 'w') as fout:
            for single_correctness_instance in single_correctness_instances:
                fout.write(json.dumps(single_correctness_instance) + '\n')

            for line in f:
                instance = json.loads(line)
                output_instance = deepcopy(instance)

                judgment = self._extract_judgment(instance['genselect_comparison'], max_idx=instance["max_idx"])
                if judgment is not None:
                    output_instance["judgment_idx"] = judgment
                else:
                    output_instance["judgment_idx"] = None
                    judgment = random.randint(0, instance["max_idx"])

                output_instance[self.cfg.output_key] = instance[f"{self.cfg.output_key}_{judgment}"]
                output_instance[self.cfg.answer_key] = instance[f"{self.cfg.answer_key}_{judgment}"]

                # Delete other variables that are not needed for the final output
                for i in range(instance["num_solutions"]):
                    # Check if the key exists before deleting
                    if f"{self.cfg.output_key}_{i}" in output_instance:
                        del output_instance[f"{self.cfg.output_key}_{i}"]
                    if f"{self.cfg.answer_key}_{i}" in output_instance:
                        del output_instance[f"{self.cfg.answer_key}_{i}"]

                for key in ["solutions", "max_idx", "num_solutions", "_full_generation"]:
                    if key in output_instance:
                        del output_instance[key]

                fout.write(json.dumps(output_instance) + '\n')


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_config')
def generate(cfg: GenSelectConfig):
    cfg = GenSelectConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = GenSelectTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenSelectConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
