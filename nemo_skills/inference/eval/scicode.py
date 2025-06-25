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

from nemo_skills.inference.eval.scicode_utils import generate_response_with_steps
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

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

    prompt_config: str = "eval/scicode/default"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_scicode_generation_config", node=SciCodeGenerationConfig)


class SciCodeGenerationTask(GenerationTask):
    def __init__(self, cfg: SciCodeGenerationConfig):
        super().__init__(cfg)

    def log_example_prompt(self, data):
        pass
        # TODO

    def generate_single_answer(self, data_point, data, is_async):
        """Will do all necessary generations to get a single answer for the data point."""
        problem_id = data_point['problem_id']
        total_steps = len(data_point['sub_steps'])
        previous_llm_code = [None] * total_steps

        prompt_template = """
PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
""".strip()

        for i in range(total_steps):
            # this comes from original implementation, not fully sure what's the reason for this if
            if (problem_id == "13" and i == 5) or (problem_id == "62" and i == 0) or (problem_id == "76" and i == 2):
                continue
            previous_llm_code = generate_response_with_steps(
                data_point, i + 1, total_steps, prompt_template, previous_llm_code, False
            )

    def llm_generate(self, data_points, data, is_async=False):
        for data_point in data_points:
            self.generate_single_answer(data_point, data, is_async)

    def get_llm_generations(self, requests_in_progress, generations):
        gen_ids = list(requests_in_progress.values())
        # outputs = self.llm.get_generations(gen_ids)

        for dp_idx in requests_in_progress.keys():
            generations[dp_idx] = {'generation': 'tmp'}

        return requests_in_progress, generations


GENERATION_TASK_CLASS = SciCodeGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_scicode_generation_config')
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
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        scicode_generation()
