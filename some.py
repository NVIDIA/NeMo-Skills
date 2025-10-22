# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_skills.inference.eval.scicode_utils import extract_python_script, prefilled_steps_code, process_problem_steps
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SciCodeGenerationConfig(GenerateSolutionsConfig):
    """SciCode benchmark generation. Will run queries multiple times including previously generated code.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "eval/scicode/background"
    with_background: bool = True

    remove_thinking: bool = True  # changing default


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_scicode_generation_config", node=SciCodeGenerationConfig)


class SciCodeGenerationTask(GenerationTask):
    def generate(self):
        print("BOOO_FROM_SCODE")


GENERATION_TASK_CLASS = SciCodeGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_scicode_generation_config")
def scicode_generation(cfg: SciCodeGenerationConfig):
    cfg = SciCodeGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SciCodeGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SciCodeGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        LOG.info("WIPP run from some.py main")
        scicode_generation()
