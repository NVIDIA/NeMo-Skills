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
import random
from collections import defaultdict
from itertools import chain
from typing import Dict, Optional

from sdp.processors.base_processor import BaseProcessor
from tqdm.contrib.concurrent import process_map

from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
from nemo_skills.utils import unroll_files

LOG = logging.getLogger(__file__)


class ReadData(BaseProcessor):

    def __init__(
        self,
        prediction_jsonl_files: Optional[str] = None,
        preprocessed_dataset_files: Optional[str] = None,
        input_key="question",
        output_key="generation",
        skip_first: int = 0,
        add_correct: bool = True,
        add_incorrect: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prediction_jsonl_files = prediction_jsonl_files
        self.preprocessed_dataset_files = preprocessed_dataset_files
        self.input_key = input_key
        self.output_key = output_key
        self.skip_first = skip_first
        self.add_correct = add_correct
        self.add_incorrect = add_incorrect

        if isinstance(self.prediction_jsonl_files, str):
            self.prediction_jsonl_files = self.prediction_jsonl_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")

        if self.prediction_jsonl_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `prediction_jsonl_files` or `preprocessed_dataset_files` should be provided")

        if not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

    def _read_preprocessed_data(self, file_handle) -> int:
        samples = []
        questions = set()
        for idx, line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            sample = json.loads(line)
            questions.add(sample[self.input_key])
            # for backward compatibility
            if self.output_key not in sample and "generated_solution" in sample:
                sample[self.output_key] = sample.pop("generated_solution")
            samples.append(sample)

        return samples

    def _parallel_read_file(self, args):
        file_path, read_fn = args
        with open(file_path, "rt", encoding="utf-8") as file_handle:
            samples = read_fn(file_handle)
        return samples

    def _read_raw_data(self, file_handle) -> int:
        samples = []

        for idx, file_line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
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

            if not self.add_correct and line_dict["is_correct"]:
                continue

            if not self.add_incorrect and not line_dict["is_correct"]:
                continue

            # for backward compatibility
            if self.output_key not in line_dict and "generated_solution" in line_dict:
                line_dict[self.output_key] = line_dict.pop("generated_solution")

            line_dict['filename'] = file_handle.name
            samples.append(line_dict)

        return samples

    def _unique_iterator(self, samples):
        seen_predictions = defaultdict(set)
        for sample in samples:
            question = sample[self.input_key]
            if sample[self.output_key] in seen_predictions[question]:
                continue

            seen_predictions[question].add(sample[self.output_key])
            yield sample

    def process(self):
        samples = []
        if self.prediction_jsonl_files:
            args = [(file, self._read_raw_data) for file in unroll_files(self.prediction_jsonl_files)]
            results = process_map(self._parallel_read_file, args, max_workers=4, chunksize=1)
            samples.extend(list(chain(*results)))
        if self.preprocessed_dataset_files:
            args = [(file, self._read_preprocessed_data) for file in unroll_files(self.preprocessed_dataset_files)]
            results = process_map(self._parallel_read_file, args, max_workers=None, chunksize=1)
            samples.extend(list(chain(*results)))
        LOG.info("Total samples before deduplication: %d", len(samples))
        samples_count = 0
        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for sample in self._unique_iterator(samples):
                fout.write(json.dumps(sample) + "\n")
                samples_count += 1
        LOG.info("Total samples after deduplication: %d", samples_count)


class GroupSamples(BaseProcessor):
    def __init__(self, group_key='input', **kwargs):
        super().__init__(**kwargs)
        self.group_key = group_key

    def process(self):
        samples = defaultdict(list)
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                sample = json.loads(line)
                samples[sample[self.group_key]].append(sample)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for groupped_samples in samples.values():
                fout.write(json.dumps(groupped_samples) + "\n")


class ShuffleAndDownsampleData(BaseProcessor):
    def __init__(
        self,
        random_seed: int,
        do_shuffle: bool,
        num_samples: Optional[int] = None,
        sampling_method: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_method = sampling_method
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.do_shuffle = do_shuffle

        if self.sampling_method not in [None, "random", "fair"]:
            raise ValueError(
                f"Sampling method {self.sampling_method} is not supported, use `None`, `random` or `fair`"
            )

        if self.sampling_method is None and self.num_samples is not None:
            raise ValueError("Number of samples can be specified only when sampling method is `random` or `fair`")

        if self.sampling_method is not None and self.num_samples is None:
            raise ValueError("Number of samples should be specified when sampling method is `random` or `fair`")

    def process(self):
        groupped_samples = []
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                samples = json.loads(line)
                groupped_samples.append(samples)

        random.seed(self.random_seed)
        if self.sampling_method is None:
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
        if self.sampling_method == "random":
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
            output_instances = output_instances[: self.num_samples]
        elif self.sampling_method == "fair":
            soln_counter = 0
            output_instances = []
            num_input_samples = sum(len(samples) for samples in groupped_samples)
            if num_input_samples < self.num_samples:
                LOG.warning(
                    "Total SFT entries %d is not less than `num_output_samples` %d, skipping downsampling.",
                    num_input_samples,
                    self.num_samples,
                )
                output_instances = list(chain(*groupped_samples))
            # downsample only if num_input_samples > self.num_samples
            while len(output_instances) < self.num_samples and num_input_samples > self.num_samples:
                for quesn_idx in range(len(groupped_samples)):
                    if len(output_instances) == self.num_samples:
                        break
                    if len(groupped_samples[quesn_idx]) > soln_counter:
                        output_instances.append(groupped_samples[quesn_idx][soln_counter])
                soln_counter += 1
            if self.do_shuffle:
                random.shuffle(output_instances)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for instance in output_instances:
                fout.write(json.dumps(instance) + "\n")


class WriteFinalSftManifest(BaseProcessor):
    def __init__(
        self,
        prompt_type: str,
        chat_format: str | None = None,  # nemotron/llama/None
        input_key: str = "input",
        output_key: str = "output",
        generation_suffix: str = "",
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_type = prompt_type
        self.input_key = input_key
        self.output_key = output_key
        self.chat_format = chat_format
        self.metadata = metadata
        self.generation_suffix = generation_suffix
        if self.generation_suffix and self.chat_format:
            raise ValueError("generation_suffix can only be used with chat_format=False")
        if not self.metadata:
            self.metadata = {}

    def process(self):
        samples_count = 0
        seen_predictions = defaultdict(set)
        with (
            open(self.input_manifest_file, "rt", encoding="utf-8") as fin,
            open(self.output_manifest_file, "wt", encoding="utf-8") as fout,
        ):
            prompt_config = get_prompt_config(self.prompt_type)
            prompt = Prompt(config=prompt_config)
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)

                question = elem[self.input_key]
                # deduplication
                if elem[self.output_key] in seen_predictions[question]:
                    continue
                seen_predictions[question].add(elem[self.output_key])

                if self.chat_format is None:
                    generation = elem.pop(self.output_key)
                    elem["input"] = prompt.build_string(input_dict=elem)
                    elem["output"] = generation + self.generation_suffix
                elif self.chat_format.lower() == "nemotron":
                    elem['conversations'] = [
                        {'value': prompt_config.user.format(**elem), 'from': 'User', 'canonical_form': ''},
                        {'value': elem.pop(self.output_key), 'from': 'Assistant', 'canonical_form': ''},
                    ]
                    elem['system'] = prompt_config.system
                    elem['mask'] = 'User'
                elif self.chat_format.lower() == "llama":
                    elem['conversations'] = [
                        {
                            'value': prompt_config.user.format(**elem),
                            'from': '<|start_header_id|>user<|end_header_id|>',
                            'canonical_form': '',
                        },
                        {
                            'value': elem.pop(self.output_key),
                            'from': '<|start_header_id|>assistant<|end_header_id|>',
                            'canonical_form': '',
                        },
                    ]
                    elem['system'] = prompt_config.system
                    elem['mask'] = '<|start_header_id|>user<|end_header_id|>'
                else:
                    raise ValueError(f"Chat format {self.chat_format} is not supported")
                elem.update(self.metadata)
                fout.write(json.dumps(elem) + "\n")
                samples_count += 1

        LOG.info("Prepared dataset size: %d", samples_count)
