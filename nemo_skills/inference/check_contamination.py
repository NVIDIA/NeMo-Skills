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

import asyncio
import json
import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CheckContaminationConfig(GenerateSolutionsConfig):
    """LLM-based check contamination parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

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
        LOG.info("Initializing CheckContaminationTask")
        super().__init__(cfg)
        LOG.info("CheckContaminationTask initialized successfully")

    def load_data(self):
        LOG.info("Loading data...")
        # Load the data as done in the base class
        data = super().load_data()
        LOG.info("Data loaded: %d data points", len(data))

        # Adjust the batch size to account for the number of similar items
        if self.cfg.top_k is None:
            self.cfg.top_k = len(data[0]["similar_items"])
            LOG.info("Set top_k to %d based on first data point", self.cfg.top_k)
        else:
            LOG.info("Using configured top_k: %d", self.cfg.top_k)

        return data

    def log_example_prompt(self, data):
        LOG.info("Logging example prompt...")
        data_point = data[0]
        query_item = data_point[self.cfg.retrieve_key]
        similar_item = data_point["similar_items"][0]
        first_element = {
            f"{self.cfg.retrieve_key}1": query_item,
            f"{self.cfg.retrieve_key}2": similar_item,
        }
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s",
            first_element,
            self.prompt.fill(first_element),
        )

    def _create_query_data(self, data_point):
        """Create query instances given the original instance"""
        LOG.info("Creating query data for data point with %d similar items", len(data_point["similar_items"]))
        query_data = []
        for similar_item in data_point["similar_items"][: self.cfg.top_k]:
            query_data.append(
                {
                    f"{self.cfg.retrieve_key}1": data_point[self.cfg.retrieve_key],
                    f"{self.cfg.retrieve_key}2": similar_item,
                }
            )

            if self.cfg.check_both_ways:
                query_data.append(
                    {
                        f"{self.cfg.retrieve_key}2": data_point[self.cfg.retrieve_key],
                        f"{self.cfg.retrieve_key}1": similar_item,
                    }
                )

        LOG.info("Created %d query instances", len(query_data))
        return query_data

    def prefill_generation(self, data_point):
        """Prefill contamination if there is a string match between the problem and the similar items"""
        LOG.info("Checking for string match prefill...")
        for similar_item in data_point["similar_items"]:
            if data_point[self.cfg.retrieve_key].strip().lower() == similar_item.strip().lower():
                LOG.info("Found exact string match - prefilling as contaminated")
                return {"generation": True}
        LOG.info("No exact string match found")
        return None

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single data point by running contamination checks on all similar items."""
        LOG.info("Processing single datapoint...")
        query_data = self._create_query_data(data_point)
        LOG.info("Created %d queries to process", len(query_data))

        # Create tasks for all queries using the semaphore-controlled generate method
        tasks = []
        for idx, query_point in enumerate(query_data):
            LOG.info("Creating task %d/%d", idx + 1, len(query_data))
            # Call the parent's process_single_datapoint which now uses _generate_with_semaphore
            tasks.append(super().process_single_datapoint(query_point, all_data))

        LOG.info("Gathering results from %d tasks...", len(tasks))
        query_results = await asyncio.gather(*tasks)
        LOG.info("All tasks completed, processing results...")

        # Process results to determine if contaminated
        all_generations = []
        contaminated = False
        for idx, result in enumerate(query_results):
            generation = result["generation"]
            all_generations.append(generation)
            LOG.info("Result %d/%d: %s", idx + 1, len(query_results), generation)
            if generation.strip() == "True":
                contaminated = True

        LOG.info("Datapoint processing complete. Contaminated: %s", contaminated)
        return {"all_generations": all_generations, "generation": contaminated}

    def postprocess(self):
        """Postprocess the output file to calculate the contamination portion."""
        LOG.info("Starting postprocessing of output file: %s", self.cfg.output_file)
        num_contaminated, total = 0, 0
        with open(self.cfg.output_file, "r", encoding="utf-8", buffering=1) as fin:
            for line in fin:
                total += 1
                data_point = json.loads(line)
                if data_point[self.cfg.generation_key]:
                    num_contaminated += 1
                if total % 100 == 0:
                    LOG.info("Processed %d lines so far...", total)

        if total > 0:
            LOG.info("Contamination portion: %.2f%% (%d/%d)", 100 * num_contaminated / total, num_contaminated, total)
        LOG.info("Postprocessing complete")


GENERATION_TASK_CLASS = CheckContaminationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_check_contamination_config")
def check_contamination(cfg: CheckContaminationConfig):
    LOG.info("Starting check_contamination with config")
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    LOG.info("Creating CheckContaminationTask...")
    task = CheckContaminationTask(cfg)
    LOG.info("Starting generation...")
    task.generate()
    LOG.info("Generation complete")


HELP_MESSAGE = get_help_message(
    CheckContaminationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()
