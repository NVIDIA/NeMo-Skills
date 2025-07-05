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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

import hydra
import openai

from nemo_skills.inference.eval.scicode_utils import extract_python_script, prefilled_steps_code, process_problem_steps
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
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
    def __init__(self, cfg: SciCodeGenerationConfig):
        super().__init__(cfg)

        if not self.use_async_loop:  # if it was True, this message is printed by base class
            LOG.info(
                "Async loop is maintaining %d generations in parallel. "
                "Use max_concurrent_requests to control the number of concurrent requests.",
                self.cfg.max_concurrent_requests,
            )
            if self.server["server_type"] in ["nemo", "megatron"] and self.prompt_template is None:
                LOG.warning(
                    "NeMo/Megatron servers don't support inflight batching, "
                    "but SciCode evaluation requires it for efficient inference. "
                    "Each request will be processed 1 by 1, which is extremely inefficient and slow! "
                    "We highly recommend switching to a server that supports inflight batching."
                )
        self.use_async_loop = True  # SciCode is a multi-call benchmark, so we have to use async loop

    def log_example_prompt(self, data):
        """Scicode is multi-call benchmark, so we can't print a single prompt."""
        return

    def generate_single_answer(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""
        problem_id = data_point['problem_id']
        total_steps = len(data_point['sub_steps'])
        previous_llm_code = [None] * total_steps
        task_solutions = {}
        total_generated_tokens = 0
        out_of_context = False
        for cur_step in range(total_steps):
            # this comes from original implementation, not fully sure what's the reason for this if
            if (problem_id, cur_step) in prefilled_steps_code:
                previous_llm_code[cur_step] = prefilled_steps_code[problem_id, cur_step]
                continue

            if out_of_context:
                task_solutions[f"{problem_id}.{cur_step}"] = '_ran_out_of_context_'
                continue

            problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
                data_point, cur_step, previous_llm_code, self.cfg.with_background
            )
            dependencies = data_point["required_dependencies"]
            assert next_step_str
            previous_code = f'{dependencies}\n{previous_code_str}\n'
            prepare_data_point = {
                'problem_steps_str': problem_steps_str,
                'next_step_str': next_step_str,
                'dependencies': dependencies,
            }
            try:
                # we want a synchronous generation here, but it will run in a thread
                llm_output = super().llm_generate([prepare_data_point], data, is_async=False)[0]
            # TODO: this is a hack (as not all servers return that),
            # but eventually we should support handling errors like this globally for all generations
            except openai.BadRequestError as e:
                if 'Please reduce the length of the messages or completion' in str(e):
                    LOG.warning(
                        "SciCode generation failed due to running out of context. "
                        "Failing for subsequent subtasks automatically.",
                    )
                    out_of_context = True
                    task_solutions[f"{problem_id}.{cur_step}"] = '_ran_out_of_context_'
                    continue
                else:
                    raise

            total_generated_tokens += llm_output.get('num_generated_tokens', 0)
            if self.cfg.remove_thinking:
                remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)
            extracted_python = extract_python_script(llm_output['generation'])
            previous_llm_code[cur_step] = extracted_python
            # TODO: save those as separate entries so that we can preserve intermediate progress on reruns
            task_solutions[f"{problem_id}.{cur_step}"] = f'{previous_code}\n{extracted_python}'

        # generation is a dict["problem_id.subtask_step": full_solution] here
        return {'generation': task_solutions, 'num_generated_tokens': total_generated_tokens}

    def llm_generate(self, data_points, data, is_async=False):
        futures = []

        with ThreadPoolExecutor(max_workers=len(data_points)) as executor:
            for data_point in data_points:
                future = executor.submit(self.generate_single_answer, data_point, data)
                futures.append(future)

        return futures

    def get_llm_generations(self, requests_in_progress, generations):
        for dp_idx, future in requests_in_progress.items():
            if future.done():
                generations[dp_idx] = future.result()
            else:
                generations[dp_idx] = {'generation': None}

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
