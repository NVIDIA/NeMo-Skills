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
import time
import logging
import sys
from dataclasses import field
from collections import defaultdict

import hydra
from tqdm import tqdm

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
    server: dict = field(default_factory=dict)

    # Override the default Generation config here
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
        self._post_init_validate_data()
        self._post_init_validate_server()
        self._post_init_validate_params()

    def _post_init_validate_data(self):
        """Validate that the data parameters adhere to the expected values"""
        if self.input_file is None:
            raise ValueError("Input file is required for checking contamination")
        if self.output_file is None:
            raise ValueError("Output file is required for checking contamination")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("code_execution", False),
        ]
        

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
        
    def _create_query_data(self, data_point):
        """Create query instances given the original instance"""
        query_data = []
        for similar_item in data_point['similar_items'][:self.cfg.top_k]:
            query_data.append(
                {
                    f'{self.cfg.retrieve_key}1': data_point[self.cfg.retrieve_key],
                    f'{self.cfg.retrieve_key}2': similar_item,
                }
            )

            if self.cfg.check_both_ways:
                query_data.append(
                    {
                        f'{self.cfg.retrieve_key}2': data_point[self.cfg.retrieve_key],
                        f'{self.cfg.retrieve_key}1': similar_item,
                    }
                )

        return query_data

    def _prefill_generation(self, data_point):
        """Prefill contamination if there is a string match between the problem and the similar items"""
        for similar_item in data_point['similar_items']:
            if data_point[self.cfg.retrieve_key].strip().lower() == similar_item.strip().lower():
                return {"generation": True}
        return None

    def sync_loop(self, data):
        """Override the sync loop to check contamination."""
        num_contaminated, total = 0, 0
        with open(self.cfg.output_file, "at", encoding="utf-8", buffering=1) as fout:
            data_points_batch = []
            for idx, data_point in tqdm(enumerate(data), total=len(data), desc="Remaining generations"):
                prefill_output = self._prefill_generation(data_point)
                if prefill_output is not None:
                    # We can bypass the LLM and directly dump the prefilled output
                    self.dump_outputs([prefill_output], [data_point], fout)
                else:
                    data_points_batch.append(data_point)

                if len(data_points_batch) == self.cfg.batch_size or idx == len(data) - 1:
                    # Create paraphrase queries for each data point
                    query_data = [
                        query_point for data_point in data_points_batch 
                        for query_point in self._create_query_data(data_point)
                    ]
                    # Get LLMs judgement on paraphrase queries for each data point
                    outputs = self.llm_generate(query_data, data)
                    output_idx = 0

                    for original_data_point in data_points_batch:
                        all_generations = []
                        elem = {}
                        contaminated = False
                        query_per_data_point = self.cfg.top_k * (2 if self.cfg.check_both_ways else 1)
                        for output in outputs[output_idx : output_idx + query_per_data_point]:
                            all_generations.append(output['generation'])
                            # If any of the generations is True, then the data point is considered contaminated
                            if output['generation'].strip() == "True":
                                contaminated = True
                                break
                        
                        output_idx += query_per_data_point
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


    def async_loop(self, data):
        """Async loop to generate generations."""

        # We first segregate the data into prefilled and non-prefilled data points
        prefilled_data_points, prefilled_outputs = [], []
        remaining_data_points = []

        for idx, data_point in enumerate(data):
            prefill_output = self._prefill_generation(data_point)
            if prefill_output is not None:
                prefilled_outputs.append(prefill_output)
                prefilled_data_points.append(data_point)
            else:
                remaining_data_points.append(data_point)

        pbar = tqdm(total=len(remaining_data_points), desc="Remaining generations")
        last_submitted_idx = 0
        requests_in_progress = {}  # generation_id -> original data_point
        data_point_idx_to_gen_ids = defaultdict(set)  # original data_point_idx -> [generation_ids]
        data_point_idx_to_all_generations = defaultdict(list)  # original data_point_idx -> [all generations]
        with open(self.cfg.output_file + "-async", "at", encoding="utf-8", buffering=1) as fout:
            # Dump prefilled data first
            if len(prefilled_data_points) > 0:
                self.dump_outputs(prefilled_outputs, prefilled_data_points, fout)

            while last_submitted_idx < len(remaining_data_points) or len(requests_in_progress) > 0:
                num_to_submit = self.cfg.max_concurrent_requests - len(requests_in_progress)
                data_points_batch = remaining_data_points[last_submitted_idx:last_submitted_idx + num_to_submit]
                if last_submitted_idx < len(remaining_data_points) and num_to_submit > 0:
                    # Create paraphrase queries for each data point
                    query_data = [
                        query_point for data_point in data_points_batch
                        for query_point in self._create_query_data(data_point)
                    ]
                    # Assert that the number of queries is correct
                    query_per_data_point = self.cfg.top_k * (2 if self.cfg.check_both_ways else 1)
                    assert (len(query_data) == len(data_points_batch) * query_per_data_point, "Query data length mismatch")

                    generation_ids = self.llm_generate(query_data, data, is_async=True)
                    for idx, gen_id in enumerate(generation_ids):
                        orig_data_point_idx = last_submitted_idx + idx // query_per_data_point
                        requests_in_progress[gen_id] = orig_data_point_idx
                        data_point_idx_to_gen_ids[orig_data_point_idx].add(gen_id)


                    last_submitted_idx += num_to_submit

                generations = self.llm.get_generations(list(requests_in_progress.keys()))

                for (gen_id, data_point_idx), gen_dict in zip(requests_in_progress.copy().items(), generations):
                    if gen_dict['generation'] is None:  # not done yet
                        continue
                    # remove the completed task from in_progress
                    requests_in_progress.pop(gen_id)
                    # Add the generation to the list of all generations for this data point
                    data_point_idx_to_all_generations[data_point_idx].append(gen_dict['generation'])
                    # Remove the generation from the set of generation ids for this data point
                    data_point_idx_to_gen_ids[data_point_idx].remove(gen_id)

                    # If all generations corresponding to this data point are done, we can dump the result
                    if len(data_point_idx_to_gen_ids[data_point_idx]) == 0:
                        # All generations for this data point are done, so we can dump the result
                        data_point_idx_to_gen_ids.pop(data_point_idx)
                        elem = {}
                        elem['all_generations'] = data_point_idx_to_all_generations[data_point_idx]
                        elem[self.cfg.generation_key] = False
                        for generation in elem['all_generations']:
                            if generation.strip() == "True":
                                elem[self.cfg.generation_key] = True
                                break

                        elem.update(remaining_data_points[data_point_idx])
                        fout.write(json.dumps(elem) + '\n')
                        
                        pbar.update(1)

                time.sleep(1)  # Prevent excessive API overload

            pbar.close()

        self.restore_async_order()


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
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()
