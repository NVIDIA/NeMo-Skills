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
    """LLM-based check contamination parameters. 
For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    input_file: str | None = None  # an output of the retrieve_similar.py script
    output_file: str | None = None  # where to save the generations

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
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
        super().__post_init__()

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
            ("sandbox", {}),
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
    
    def llm_generate(self, data_points, data, is_async=False):
        """Override the base class method to create a 1:N mapping between data points and contamination queries."""
        # Create the query instances per data point
        query_data_batch = [
            query_point 
            for data_point in data_points 
            for query_point in self._create_query_data(data_point)
        ]
        
        # Get the LLM judgement on the queries
        outputs = super().llm_generate(query_data_batch, data, is_async)
        
        # Postprocessing of outputs to create a N:1 mapping between contamination results and data points
        query_per_data_point = self.cfg.top_k * (2 if self.cfg.check_both_ways else 1)
        
        if not is_async:
            output_idx = 0
            proc_outputs = []
            for _ in data_points:
                proc_output = {"all_generations": [], "generation": False}
                for output in outputs[output_idx : output_idx + query_per_data_point]:
                    proc_output["all_generations"].append(output['generation'])
                    # If any of the generations is True, then the data point is considered contaminated
                    if output['generation'].strip() == "True":
                        proc_output["generation"] = True
                        
                proc_outputs.append(proc_output)
                output_idx += query_per_data_point

            return proc_outputs
        else:
            # Club the generation IDs into tuples for each data point
            generation_ids = []
            for idx in range(0, len(outputs), query_per_data_point):
                # We mark each generation ID as "Unfinished" to indicate that it is not done yet
                generation_ids.append(outputs[idx: idx + query_per_data_point])

            return generation_ids

    def get_llm_generations(self, requests_in_progress, generations):
        """Override the base class method to synchronize the N:1 mapping between contamination results and original data points. This is done by getting the LLM generations for all the queries for each data point and then processing the "generation" key.
        
        requests_in_progress: A dictionary of the form {original_data_point_idx: gen_id_list}
        generations: A dictionary of the form {original_data_point_idx: gen_dict}
        """
        for original_dp_idx, gen_id_list in requests_in_progress.items():
            # Get the LLM generations for all the queries remaining for this data point
            gen_dict_list = self.llm.get_generations(gen_id_list)
            # If all generations have finished, we can process the "generation" key, otherwise keep it as None
            rem_gen_id_list = []
            for gen_id, gen_dict in zip(gen_id_list, gen_dict_list):
                if gen_dict['generation'] is None:
                    # If any of the generations is not done, we cannot process the data point
                    rem_gen_id_list.append(gen_id)
                else:
                    if "all_generations" not in generations[original_dp_idx]:
                        generations[original_dp_idx]['all_generations'] = []

                    generations[original_dp_idx]['all_generations'].append(gen_dict['generation'])
                    
            # Update the requests in progress to include the remaining generation IDs
            requests_in_progress[original_dp_idx] = rem_gen_id_list

            if rem_gen_id_list:
                # There are still generations to be processed
                generations[original_dp_idx]["generation"] = None
            else:
                # All generations have finished
                # If any of the generations is True, then the data point is considered contaminated
                contaminated = any([generation.strip() == "True" for generation in generations[original_dp_idx]['all_generations']])
                generations[original_dp_idx]['generation'] = contaminated

        return (requests_in_progress, generations)

    def postprocess(self):
        """Postprocess the output file to calculate the contamination portion."""
        num_contaminated, total = 0, 0
        with open(self.cfg.output_file, "r", encoding="utf-8", buffering=1) as fin:
            for line in fin:
                total += 1
                data_point = json.loads(line)
                if data_point[self.cfg.generation_key]:
                    num_contaminated += 1

        if total > 0:
            LOG.info("Contamination portion: %.2f%% (%d/%d)", 100 * num_contaminated / total, num_contaminated, total)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_check_contamination_config')
def check_contamination(cfg: CheckContaminationConfig):
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = CheckContaminationTask(cfg)
    task.generate()


HELP_MESSAGE = (
    CheckContaminationConfig.__doc__ + "\n\n" + 
    get_help_message(
        CheckContaminationConfig,
        server_params=server_params(), 
    )
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()
