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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.prompt.utils import Prompt, PromptConfig, datasets, prompt_types
from nemo_skills.inference.server.model import get_model, server_params
from nemo_skills.utils import get_help_message, setup_logging

LOG = logging.getLogger(__file__)


@dataclass
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 512
    repetition_penalty: float = 1.0


@dataclass
class GenerateSolutionsConfig:
    """Top-level parameters for the script"""

    output_file: str  # Where to save the generations
    # Inference server configuration {server_params}
    server: dict
    # Sandbox configuration {sandbox_params}
    sandbox: dict
    # Prompt configuration.
    # Available pre-configured prompts: {prompt_types}.
    prompt: PromptConfig = field(default_factory=PromptConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    # Choices: {datasets}.
    dataset: Optional[str] = None
    split_name: Optional[str] = None  # Can be train, validation, test or train_full (train + validation)
    data_file: Optional[str] = None  # Can directly specify a data file, if using a custom dataset

    batch_size: int = 16
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file
    # if > 0, will skip this many samples from the beginning of the data file.
    # Useful if need to run multiple slurm jobs on the same data file
    offset: int = 0

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if self.data_file is not None:
            if self.dataset is not None or self.split_name is not None:
                raise ValueError("Either `data_file` or `dataset` and `split_name` should be provided, but not both")
        else:
            if self.dataset is None or self.split_name is None:
                raise ValueError("Either `data_file` or `dataset` and `split_name` should be provided")
            self.data_file = Path(__file__).parents[2] / "datasets" / self.dataset / f"{self.split_name}.jsonl"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


@hydra.main(version_base=None, config_name='generation_config', config_path='.')
def generate_solutions(cfg: GenerateSolutionsConfig):
    cfg = OmegaConf.to_object(cfg)

    LOG.info("Config used: %s", cfg)
    sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
    llm = get_model(**cfg.server, sandbox=sandbox)

    # making sure output folder exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.data_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))

    # skipping based on the offset first
    data = data[cfg.offset :]

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")

    # additionally, skipping whatever is pre-filled, assuming offset didn't change
    data = data[starting_idx:]

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        prompts = []
        data_points = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx == cfg.max_samples:
                break

            prompts.append(Prompt(cfg.prompt, data_point))
            data_points.append(data_point)

            if len(prompts) == cfg.batch_size:
                # batch-computing the outputs
                outputs = llm(stop_phrases=list(cfg.prompt.stop_phrases), prompts=prompts, **asdict(cfg.inference))
                for output, original_data_point in zip(outputs, data_points):
                    # to make it easier to follow up with evaluation and limit accidental errors, we are adding
                    # all of the ground-truth data to the output file alongside the generated solutions
                    output.update(original_data_point)
                    fout.write(json.dumps(output) + "\n")
                prompts = []
                data_points = []

        # collecting the final batch
        if len(prompts) > 0:
            outputs = llm(stop_phrases=list(cfg.prompt.stop_phrases), prompts=prompts, **asdict(cfg.inference))
            for output, original_data_point in zip(outputs, data_points):
                output.update(original_data_point)
                fout.write(json.dumps(output) + "\n")


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    datasets=datasets,
    prompt_types=prompt_types,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate_solutions()
