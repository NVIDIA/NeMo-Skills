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

import hydra

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_reward_model, server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class RewardModelConfig(GenerateSolutionsConfig):
    """LLM reward model parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

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

    @classmethod
    def get_server_command_fn(cls) -> callable:
        from nemo_skills.pipeline.utils import get_reward_server_command

        return get_reward_server_command


GENERATION_TASK_CLASS = RewardModelTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_reward_model_config')
def score(cfg: RewardModelConfig):
    cfg = RewardModelConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = RewardModelTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    RewardModelConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        score()
