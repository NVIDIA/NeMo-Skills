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
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import tqdm
import yaml
from omegaconf import MISSING, OmegaConf

sys.path.append(str(Path(__file__).absolute().parents[2]))

from nemo_skills.finetuning.filtering_utils import downsample_data, process_bad_solutions
from nemo_skills.inference.prompt.utils import PromptConfig, get_prompt
from nemo_skills.utils import get_help_message, setup_logging, unroll_files

LOG = logging.getLogger(__file__)

# TODO: this should be done as a pipeline with different filterings / downsampling
#       ideally directly use SDP for this


def get_default_prompt_config():
    # by default reading code_sfted.yaml, but users can override
    with open(
        Path(__file__).parents[1] / "inference" / "prompt" / f"code_sfted.yaml",
        "rt",
        encoding="utf-8",
    ) as fin:
        prompt_config = PromptConfig(**yaml.safe_load(fin))
    prompt_config.context_type = "empty"
    prompt_config.examples_type = "gsm8k_text_with_code"  # not used since num_few_shots = 0
    prompt_config.num_few_shots = 0
    return prompt_config


@dataclass
class PrepareSFTDataConfig:
    """Top-level parameters for the script"""

    prediction_jsonl_files: Optional[str] = None  # can specify multiple patters separated by space
    preprocessed_dataset_files: Optional[str] = None  # can specify datasets from HF instead of prediction_jsonl_files
    output_path: str = MISSING
    # can provide additional metadata to store (e.g. dataset or generation_type)
    metadata: Dict[Any, Any] = field(default_factory=dict)
    skip_first: int = 0  # useful for skipping validation set from train_full generation (it's always first)
    add_correct: bool = True  # can set to False if only want to export incorrect solutions
    add_incorrect: bool = False  # if True, saves only incorrect solutions instead of correct

    downsampling_method: Optional[str] = None  # random or fair
    num_output_samples: int = -1

    prompt: PromptConfig = field(default_factory=get_default_prompt_config)

    filters: List[str] = field(default_factory=lambda: ["multi_boxed", "broken_code"])
    text_filter_type: Optional[str] = None
    trim_solutions: bool = False

    def __post_init__(self):
        """Building data_file from dataset/split_name if not provided directly."""
        if self.prediction_jsonl_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `prediction_jsonl_files` or `preprocessed_dataset_files` should be provided")

        if self.downsampling_method is None and self.num_output_samples != -1:
            raise ValueError("Cannot use `num_output_samples` without `downsampling_method`")

        if self.downsampling_method is not None and self.num_output_samples == -1:
            raise ValueError("Cannot use `downsampling_method` with `num_output_samples = -1`")

        if not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")


def read_preprocessed_data(file_paths, grouped_samples: Dict[str, List]):
    for file_path in file_paths:
        with open(file_path, "rt", encoding="utf-8") as file_handle:
            for line in tqdm.tqdm(file_handle):
                sample = json.loads(line)
                grouped_samples[sample["question"]].append(sample)


def read_raw_data(file_handles, cfg: PrepareSFTDataConfig, grouped_samples: Dict[str, List]):
    data_size = 0
    for idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
        if idx < cfg.skip_first:
            continue
        data_size += 1

        seen_predictions = set()
        for file_line in lines:
            # if different files have different number of lines
            if file_line is None:
                continue
            line_dict = json.loads(file_line)
            # can be empty for incomplete generations
            if not line_dict:
                continue
            # skipping any incomplete generations
            if "is_correct" not in line_dict:
                LOG.warning("Found incomplete generations (is_correct field is missing) - skipping")
                continue

            if not cfg.add_correct and line_dict["is_correct"]:
                continue

            if not cfg.add_incorrect and not line_dict["is_correct"]:
                continue

            if line_dict["generated_solution"] in seen_predictions:
                continue
            seen_predictions.add(line_dict["generated_solution"])
            grouped_samples[line_dict["question"]].append(line_dict)

    return data_size


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_prepare_sft_data_config", node=PrepareSFTDataConfig)


@hydra.main(version_base=None, config_name="base_prepare_sft_data_config")
def prepare_sft_data(cfg: PrepareSFTDataConfig):
    cfg = OmegaConf.to_object(cfg)
    LOG.info("Config used: %s", cfg)

    data_size = None
    grouped_samples = defaultdict(list)
    if cfg.preprocessed_dataset_files:
        read_preprocessed_data(cfg.preprocessed_dataset_files, grouped_samples)

    if cfg.prediction_jsonl_files:
        file_handles = [
            open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(cfg.prediction_jsonl_files)
        ]
        data_size = read_raw_data(file_handles, cfg, grouped_samples)
        for handle in file_handles:
            handle.close()

    prepared_data = []
    total_covered = 0
    samples_per_question = []
    for question, samples in tqdm.tqdm(grouped_samples.items()):
        filtered_solutions = process_bad_solutions(samples, cfg.filters, cfg.text_filter_type, cfg.trim_solutions)

        samples_per_question.append(len(filtered_solutions))
        total_covered += samples_per_question[-1] != 0

        for sample in filtered_solutions:
            # including all fields in case they are useful for training
            elem = sample.copy()
            # NeMo requires input/output fields
            elem["input"] = get_prompt(cfg.prompt, {"question": question})
            elem["output"] = elem.pop("generated_solution")
            elem.update(cfg.metadata)
            prepared_data.append(elem)

    samples_per_question = np.array(samples_per_question)
    LOG.info("Total SFT entries: %d", len(prepared_data))
    if data_size is not None:
        LOG.info("Dataset coverage: %.2f%%", 100 * total_covered / data_size)
    LOG.info("Samples per question = %.2f ± %.2f", samples_per_question.mean(), samples_per_question.std())
    random.shuffle(prepared_data)

    if cfg.downsampling_method is not None:
        if cfg.num_output_samples >= len(prepared_data):
            LOG.warning(
                "Total SFT entries %d is not less than `cfg.num_output_samples` %d, skipping downsampling.",
                len(prepared_data),
                cfg.num_output_samples,
            )
        else:
            prepared_data = downsample_data(prepared_data, cfg.downsampling_method, cfg.num_output_samples)
            LOG.info("Total SFT entries after downsampling: %d", len(prepared_data))

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)
    with open(cfg.output_path, "wt", encoding="utf-8") as fout:
        for elem in tqdm.tqdm(prepared_data):
            fout.write(json.dumps(elem) + "\n")


if __name__ == "__main__":
    if '--help' in sys.argv:
        help_msg = get_help_message(PrepareSFTDataConfig)
        print(help_msg)
    else:
        setup_logging()
        prepare_sft_data()
