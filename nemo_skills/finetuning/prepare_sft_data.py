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
from dataclasses import field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import tqdm
from omegaconf import MISSING

sys.path.append(str(Path(__file__).absolute().parents[2]))

from nemo_skills.finetuning.filtering_utils import downsample_data, process_bad_solutions
from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)

# TODO: this should be done as a pipeline with different filterings / downsampling
#       ideally directly use nemo curator for this


@nested_dataclass
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

    # TODO: support prompt config directly, but there are some hydra issues
    prompt_type: str = 'openmathinstruct/sft'

    filters: List[str] = field(default_factory=lambda: ["multi_boxed", "broken_code"])
    text_filter_type: Optional[str] = None
    trim_solutions: bool = False

    chat_format: bool = False  # whether to use NeMo's chat format

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


def read_preprocessed_data(file_paths, grouped_samples: Dict[str, List]) -> int:
    questions = set()
    for file_path in file_paths:
        with open(file_path, "rt", encoding="utf-8") as file_handle:
            for line in tqdm.tqdm(file_handle):
                sample = json.loads(line)
                questions.add(sample["question"])
                # for backward compatibility
                if "generation" not in sample and "generated_solution" in sample:
                    sample["generation"] = sample.pop("generated_solution")
                grouped_samples[sample["question"]].append(sample)

    return len(questions)


def read_raw_data(file_handles, cfg: PrepareSFTDataConfig, grouped_samples: Dict[str, List]) -> int:
    questions = set()
    for idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
        if idx < cfg.skip_first:
            continue

        seen_predictions = {}
        for lidx, file_line in enumerate(lines):
            # if different files have different number of lines
            if file_line is None:
                continue
            line_dict = json.loads(file_line)
            # can be empty for incomplete generations
            if not line_dict:
                continue

            questions.add(line_dict["question"])
            if line_dict["question"] not in seen_predictions:
                seen_predictions[line_dict["question"]] = set()

            # skipping any incomplete generations
            if "is_correct" not in line_dict:
                LOG.warning("Found incomplete generations (is_correct field is missing) - skipping")
                continue

            if not cfg.add_correct and line_dict["is_correct"]:
                continue

            if not cfg.add_incorrect and not line_dict["is_correct"]:
                continue

            # for backward compatibility
            if "generation" not in line_dict and "generated_solution" in line_dict:
                line_dict["generation"] = line_dict.pop("generated_solution")

            if line_dict["generation"] in seen_predictions[line_dict["question"]]:
                continue

            seen_predictions[line_dict["question"]].add(line_dict["generation"])
            line_dict['filename'] = file_handles[lidx].name
            grouped_samples[line_dict["question"]].append(line_dict)

    return len(questions)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_prepare_sft_data_config", node=PrepareSFTDataConfig)


@hydra.main(version_base=None, config_name="base_prepare_sft_data_config")
def prepare_sft_data(cfg: PrepareSFTDataConfig):
    cfg = PrepareSFTDataConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    data_size = 0
    grouped_samples = defaultdict(list)
    if cfg.preprocessed_dataset_files:
        data_size += read_preprocessed_data(cfg.preprocessed_dataset_files, grouped_samples)

    if cfg.prediction_jsonl_files:
        file_handles = [
            open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(cfg.prediction_jsonl_files)
        ]
        data_size += read_raw_data(file_handles, cfg, grouped_samples)
        for handle in file_handles:
            handle.close()

    prepared_data = []
    total_covered = 0
    samples_per_question = []
    prompt_config = get_prompt_config(cfg.prompt_type)
    prompt = Prompt(config=prompt_config)
    # only looping over the correct samples (unless asked for incorrect)
    for question, samples in tqdm.tqdm(grouped_samples.items()):
        filtered_solutions = process_bad_solutions(samples, cfg.filters, cfg.text_filter_type, cfg.trim_solutions)
        samples_per_question.append(len(filtered_solutions))
        total_covered += samples_per_question[-1] != 0

        for sample in filtered_solutions:
            # including all fields in case they are useful for training
            elem = sample.copy()
            if cfg.chat_format:
                elem['conversations'] = [
                    {'value': question, 'from': 'User', 'canonical_form': ''},
                    {'value': elem.pop("generation"), 'from': 'Assistant', 'canonical_form': ''},
                ]
                elem['system'] = prompt_config.system
                elem['mask'] = 'User'
                elem['type'] = None
            else:
                elem["input"] = prompt.build_string(input_dict={"question": question})
                elem["output"] = elem.pop("generation")
            elem.update(cfg.metadata)
            prepared_data.append(elem)

    samples_per_question = np.array(samples_per_question)
    # adding questions that don't have any samples in grouped_samples
    if data_size > len(samples_per_question):
        samples_per_question = np.concatenate(
            [samples_per_question, np.zeros(data_size - len(samples_per_question), dtype=int)]
        )
    LOG.info("Total SFT entries: %d", len(prepared_data))
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
    if '--help' in sys.argv or '-h' in sys.argv:
        help_msg = get_help_message(PrepareSFTDataConfig)
        print(help_msg)
    else:
        setup_logging()
        prepare_sft_data()
