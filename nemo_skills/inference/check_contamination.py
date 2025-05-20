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

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CheckContaminationConfig(GenerateSolutionsConfig):
    """Top-level parameters for the script"""

    input_file: str | None = None  # an output of the retrieve_similar.py script
    output_file: str | None = None  # where to save the generations

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)

    # Override the default Generation config here
    # Async generation requires non-trivial work to support. We will not support it for now.
    # Since contamination is a fast operation, we can afford to do it synchronously
    use_async_loop: bool = False
    code_execution: bool = False
    prompt_config: str = "judge/check-contamination"
    generation_key: str = "contaminated"

    # Contamination-specific parameters
    retrieve_key: str = "problem"  # will be used to fill in prompt with retrieve_key1 and retrieve_key2
    # ask both with retrieve_key1 / retrieve_key2 and retrieve_key2 / retrieve_key1 and fill True if any is True
    check_both_ways: bool = False
    # Number of similar items to check. If not provided, will use the number of similar items in the first data point.
    top_k: int | None = None

    def __post_init__(self):
        if self.input_file is None:
            raise ValueError("Input file is required for checking contamination")
        if self.output_file is None:
            raise ValueError("Output file is required for checking contamination")

        self._post_init_validate_server()
        self._post_init_validate_params()

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.use_async_loop:
            raise ValueError("Async generation is not supported for checking contamination")
        if self.code_execution:
            raise ValueError("Code execution is not supported for checking contamination")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_check_contamination_config", node=CheckContaminationConfig)


class CheckContaminationTask(GenerationTask):
    def __init__(self, cfg: CheckContaminationConfig):
        super().__init__(cfg)

    def load_data(self):
        # Load the data as done in the base class
        data = super().load_data()

        # Adjust the batch size to account for the number of similar items
        if self.cfg.top_k is None:
            self.cfg.top_k = len(data[0]['similar_items'])
        self.cfg.batch_size = max(1, self.cfg.batch_size // self.cfg.top_k // (2 if self.cfg.check_both_ways else 1))

        return data

    def log_example_prompt(self, data):
        data_point = data[0]
        query_item = data_point[self.cfg.retrieve_key]
        similar_item = data_point['similar_items'][0]
        first_element = {
            f'{self.cfg.retrieve_key}1': query_item,
            f'{self.cfg.retrieve_key}2': similar_item,
        }
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s",
            first_element,
            self.prompt.fill(first_element),
        )

    def sync_loop(self, data):
        """Override the sync loop to check contamination."""
        num_contaminated, total = 0, 0
        with open(self.cfg.output_file, "at", encoding="utf-8", buffering=1) as fout:
            data_points_batch = []
            for idx, data_point in tqdm(enumerate(data), total=len(data), desc="Remaining generations"):
                data_points_batch.append(data_point)
                if len(data_points_batch) == self.cfg.batch_size or idx == len(data) - 1:
                    query_data = []
                    for original_data_point in data_points_batch:
                        for similar_item in original_data_point['similar_items']:
                            query_data.append(
                                {
                                    f'{self.cfg.retrieve_key}1': original_data_point[self.cfg.retrieve_key],
                                    f'{self.cfg.retrieve_key}2': similar_item,
                                }
                            )

                            if self.cfg.check_both_ways:
                                query_data.append(
                                    {
                                        f'{self.cfg.retrieve_key}2': original_data_point[self.cfg.retrieve_key],
                                        f'{self.cfg.retrieve_key}1': similar_item,
                                    }
                                )

                    outputs = self.llm_generate(query_data, data)
                    output_idx = 0
                    for original_data_point in data_points_batch:
                        all_generations = []
                        elem = {}
                        contaminated = False
                        for output in outputs[
                            output_idx : output_idx + self.cfg.top_k * (2 if self.cfg.check_both_ways else 1)
                        ]:
                            all_generations.append(output['generation'])
                            if output['generation'].strip() == "True":
                                contaminated = True
                            output_idx += 1
                        elem[self.cfg.generation_key] = contaminated
                        if contaminated:
                            num_contaminated += 1
                        total += 1
                        elem["all_generations"] = all_generations
                        for key in elem:
                            original_data_point.pop(key, None)
                        elem.update(original_data_point)
                        fout.write(json.dumps(elem) + '\n')

        if total > 0:
            LOG.info("Contamination portion: %.2f%% (%d/%d)", 100 * num_contaminated / total, num_contaminated, total)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_check_contamination_config')
def check_contamination(cfg: CheckContaminationConfig):
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = CheckContaminationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    CheckContaminationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()
