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
from pathlib import Path

import hydra
from tqdm import tqdm

from nemo_skills.inference.server.code_execution_model import server_params
from nemo_skills.inference.server.reward_model import get_reward_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class RewardModelConfig:
    """LLM reward model parameters."""

    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str  # we will fetch it from dataset dir if not provided

    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset
    output_file: str | None = None  # Where to save the generations if `input_file` is provided
    # Can specify an input directory, where the file will be inferred output.jsonl if no seed
    # is provided, and output-rs{{seed}}.jsonl. This pattern is used to match the output files from
    # the `generate` pipeline
    input_dir: str | None = None
    # Where to save the generations (with the identical file name) if `input_dir` is provided
    output_dir: str | None = None
    # Used to identify the input file if `input_dir` is provided. If `random_seed` is not provided,
    # the input will be assumed to be from 'greedy' generation
    random_seed: str | None = None
    batch_size: int = 128
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # if > 0, will skip this many samples from the beginning of the data file.
    # Useful if need to run multiple slurm jobs on the same data file
    offset: int = 0

    reward_model_score_key: str = "reward_model_score"

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    def __post_init__(self):
        if self.random_seed.strip() == 'None':
            self.random_seed = None
        if self.input_file is None and self.input_dir is not None:
            seed = f'-rs{self.random_seed}' if self.random_seed is not None else ''
            self.input_file = Path(self.input_dir) / f"output{seed}.jsonl"
            self.output_file = Path(self.output_dir) / f"output{seed}.jsonl"
        elif self.input_file is not None and self.input_dir is None:
            if self.output_file is None:
                raise ValueError("Output file should be provided if providing `input_file`")
        else:
            raise ValueError("`input_file` and `input_dir` cannot be provided at the same time")

        if self.server["server_type"] != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server["server_type"] == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reward_model_config", node=RewardModelConfig)


@hydra.main(version_base=None, config_name='base_reward_model_config')
def generate(cfg: RewardModelConfig):
    cfg = RewardModelConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)
    llm = get_reward_model(**cfg.server)

    # making sure output dir exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.input_file, "rt", encoding="utf-8") as fin:
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

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template)
    LOG.info("Prompt used: %s", prompt)

    # need to account for anything that's prefilled
    if 0 <= cfg.max_samples <= starting_idx:
        cfg.max_samples = 0

    if starting_idx < cfg.max_samples:
        cfg.max_samples -= starting_idx

    if cfg.max_samples < 0 or cfg.max_samples > len(data):
        cfg.max_samples = len(data)

    if len(data) == 0:  # we might not have any examples if skip_filled=True
        return

    LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", data[0], prompt.fill(data[0]))

    if cfg.dry_run:
        return

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        data_points = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx >= cfg.max_samples:
                break
            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == cfg.max_samples - 1:
                outputs = llm.score(
                    prompts=[prompt.fill(dp, include_generation=True) for dp in data_points],
                )

                for output, original_data_point in zip(outputs, data_points):
                    # to make it easier to follow up with evaluation and limit accidental errors, we are adding
                    # all of the ground-truth data to the output file alongside the generated solutions
                    result = output.pop('reward_model_score')
                    output[cfg.reward_model_score_key] = result
                    original_data_point.pop(cfg.reward_model_score_key, None)
                    output.update(original_data_point)
                    fout.write(json.dumps(output) + "\n")
                data_points = []


HELP_MESSAGE = get_help_message(
    RewardModelConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
