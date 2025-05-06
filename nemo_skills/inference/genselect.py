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
import re
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path
from copy import deepcopy
import random

import hydra
import typer
from typing import Any
from tqdm import tqdm
from os import path, makedirs
from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.inference.generate import InferenceConfig, GenerationTask
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class GenSelectConfig:
    """Genselect parameters."""

    input_dir: str  # Where to save the generations
    output_dir: str  # Where to save the generations
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str = "openmath/genselect"  # we will fetch it from dataset dir if not provided
    prefix_generation_to_response: bool = False  # whether to include "generation" as prefix to the response
    # if True, model will be prompted to continue "generation" without closing assistant tag
    continue_prefix_generation: bool = False

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    dataset: str | None = None
    split: str | None = None  # Generally one of train/test, but can be anything since it's used as part of a file name
    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset

    batch_size: int = 128
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    max_concurrent_requests: int = 1024  # Maximum number of concurrent requests to the server for the async loop
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    generation_key: str = "gen_rm_comparison"
    # if specified, we will have a loop over that key in the data file and
    # treat each element as a new turn of conversation
    # E.g. if multi_turn_key="turns" and a line in your data file has
    # turns: ['Hey how are you?', 'And where do you live?']
    # the generations will also be a list with the first entry corresponding to prompt
    # with the first question, second entry to both first question, first answer and second question
    # and so on
    multi_turn_key: str | None = None

    # set to False if you want to use synchronous loop instead of async. Async loop means we will send all
    # data to engine at the same time (batch size is ignored) and then write the output as soon as it's ready
    # to `output_file`-async (and put it back in order after all generations are done)
    use_async_loop: bool = True
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # extra stop phrases for llms
    extra_stop_phrases: list[str] = field(default_factory=list)

    examples_type: str | None = None  # to be able to customize few-shot examples
    sandbox: dict = field(default_factory=dict)
    code_execution: bool = False
    # Controls how many code executions are allowed in prompt (useful for models that support dynamically setting this)
    # if total_code_executions placeholder is not in the prompt, this parameter has no effect
    # Can be int, (min,max) tuple, or None
    # If (min,max) tuple, will be randomly sampled from random.randint(min_val, max_val) for each sample in a batch
    # useful to generate data with variable number of total_code_executions_in_prompt
    total_code_executions_in_prompt: Any = None
    # When True, total_code_executions_in_prompt override model defaults
    override_max_code_executions: bool = False


    def __post_init__(self):
        if self.inference.random_seed is None:
            raise ValueError("Random seed is required for genselect")
        self.input_file = str(Path(self.input_dir) / f"output-rs{self.inference.random_seed}.jsonl")
        self.output_file = str(Path(self.output_dir) / "comparison_judgment" / f"output-rs{self.inference.random_seed}.jsonl")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        if self.server["server_type"] != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server["server_type"] == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_config", node=GenSelectConfig)


class GenSelectTask(GenerationTask):
    def __init__(self, cfg: GenSelectConfig):
        super().__init__(cfg)

    def get_generation_module(self):
        return "nemo_skills.inference.genselect"

    def _extract_judgment(self, generation, max_idx=None):
        """Extract the judgment from the generation."""
        judgment = None

        try:
            matches = re.findall(r"Judg[e]?ment: (\d+)", generation)
            # print(matches)

            if matches:
                number = matches[-1]
                judgment = int(number)
                if max_idx is not None and judgment > max_idx:
                    judgment = None
            else:
                judgment = None

        except:
            judgment = None

        if judgment is not None and max_idx is not None:
            if judgment > max_idx:
                judgment = None

        return judgment
    

    def postprocess(self):
        single_answer_instances_file = path.join(self.input_dir, "single_answer_instances.jsonl")
        single_answer_instances = [json.loads(line) for line in open(single_answer_instances_file, "r")]

        input_file = self.output_file
        output_file = Path(self.output_dir) / f"output-rs{self.inference.random_seed}.jsonl"

        with open(input_file, 'r') as f, open(output_file, 'w') as fout:
            for single_answer_instance in single_answer_instances:
                fout.write(json.dumps(single_answer_instance) + '\n')

            for line in f:
                instance = json.loads(line)
                output_instance = deepcopy(instance)
                output_instance = {
                    "problem": instance['problem'], 
                    "expected_answer": instance['expected_answer']
                }

                judgment = self._extract_judgment(instance['gen_rm_comparison'], max_idx=instance["max_idx"])
                if judgment:
                    output_instance["judgment_idx"] = judgment
                else:
                    output_instance["judgment_idx"] = None
                    judgment = random.randint(0, instance["max_idx"])

                output_instance["predicted_answer"] = instance[f'predicted_answer_{judgment}']
                output_instance["is_correct"] = instance[f'is_correct_{judgment}']

                fout.write(json.dumps(output_instance) + '\n')


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_config')
def generate(cfg: GenSelectConfig):
    cfg = GenSelectConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = GenSelectTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenSelectConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
