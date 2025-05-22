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
from pathlib import Path

import hydra

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.server.reward_model import get_reward_model
from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class RewardModelConfig(GenerateSolutionsConfig):
    """LLM reward model parameters. 
For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

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

    # Async loop is currently not supported for reward model
    # Currently reward models are quite fast, so we don't need to use async loop
    use_async_loop: bool = False
    # Code execution is not supported for reward model
    code_execution: bool = False

    # Generation is used to construct the prompt for the reward model
    prefix_generation_to_response: bool = True

    # Key to store the reward model score
    generation_key: str = "reward_model_score"
    # Reward model specific parameters
    reward_model_type: str = "orm"

    def __post_init__(self):
        super().__post_init__()

    def _post_init_validate_data(self):
        if self.random_seed.strip() == 'None':
            self.random_seed = None
        if self.input_file is None and self.input_dir is not None:
            seed = f'-rs{self.random_seed}' if self.random_seed is not None else ''
            self.input_file = Path(self.input_dir) / f"output{seed}.jsonl"
            self.output_file = Path(self.output_dir) / f"output{seed}.jsonl"
        elif self.input_file is not None and self.input_dir is None:
            if self.output_file is None:
                raise ValueError("Output file should be provided if providing `input_file`")
        else:
            raise ValueError("`input_file` and `input_dir` cannot be provided at the same time")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("use_async_loop", False),
            ("code_execution", False),
            ("sandbox", {}),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reward_model_config", node=RewardModelConfig)


class RewardModelTask(GenerationTask):
    def __init__(self, cfg: RewardModelConfig):
        super().__init__(cfg)

    def setup_llm(self):
        """LLM is a reward model"""
        return get_reward_model(model_type=self.cfg.reward_model_type, **self.cfg.server)

    def llm_generate(self, data_points, data):
        """Rather than generating, we are scoring the data points"""
        outputs = self.llm.score(prompts=[self.prompt.fill(dp, data) for dp in data_points])
        return outputs


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_reward_model_config')
def score(cfg: RewardModelConfig):
    cfg = RewardModelConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = RewardModelTask(cfg)
    task.generate()


HELP_MESSAGE = (
    RewardModelConfig.__doc__ + "\n\n" + 
    get_help_message(
        RewardModelConfig,
        server_params=server_params(), 
    )
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        score()
