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
from os import path

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, prefill_judgement, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# TODO: should we move slightly confusing input/output dir and rs to the pipeline wrapper?


@nested_dataclass(kw_only=True)
class LlmMathJudgeConfig(GenerateSolutionsConfig):
    """Top-level parameters for the script"""

    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset
    output_file: str | None = None  # Where to save the generations if `input_file` is provided
    # Can specify an input directory, where the file will be inferred output.jsonl if no seed
    # is provided, and output-rs{{seed}}.jsonl. This pattern is used to match the output files from
    # the `generate` pipeline
    input_dir: str | None = None
    # Where to save the generations (with the identical file name) if `input_dir` is provided
    output_dir: str | None = None
    # Used to identify the input file if `input_dir` is provided. If `random_seed` is not provided,
    # the input will be assumed to be from 'greedy' generation
    random_seed: str | None = None
    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)

    # Override the default Generation config here
    prompt_config: str = "judge/math"
    generation_key: str = "judgement"

    def __post_init__(self):
        self._post_init_validate_data()
        self._post_init_validate_server()
        self._post_init_validate_params()

    def _post_init_validate_data(self):
        if self.random_seed.strip() == 'None':
            self.random_seed = None
        if self.input_file is None and self.input_dir is not None:
            seed = f'-rs{self.random_seed}' if self.random_seed is not None else ''
            self.input_file = path.join(self.input_dir, f"output{seed}.jsonl")
            self.output_file = path.join(self.output_dir, f"output{seed}.jsonl")
        elif self.input_file is not None and self.input_dir is None:
            if self.output_file is None:
                raise ValueError("Output file should be provided if providing `input_file`")
        else:
            raise ValueError("`input_file` and `input_dir` cannot be provided at the same time")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_llm_math_judge_config", node=LlmMathJudgeConfig)


class LLMMathJudgeTask(GenerationTask):
    def __init__(self, cfg: LlmMathJudgeConfig):
        super().__init__(cfg)

    def preprocess_data(self, data):
        """Extract the predicted answer from the generation."""
        for data_point in data:
            if "predicted_answer" not in data_point:
                data_point["predicted_answer"] = extract_answer(data_point["generation"])

        return data

    def prefill_generation(self, data_point):
        """Prefill judgement"""
        judgement = prefill_judgement(data_point)
        if judgement is None:
            return None
        else:
            return {"generation": judgement}


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_llm_math_judge_config')
def generate(cfg: LlmMathJudgeConfig):
    cfg = LlmMathJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = LLMMathJudgeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    LlmMathJudgeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
