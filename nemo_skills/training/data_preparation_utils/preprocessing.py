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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from typing import Dict, Optional

from sdp.processors.base_processor import BaseProcessor
from tqdm.contrib.concurrent import process_map

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import unroll_files

LOG = logging.getLogger(__file__)


class ReadData(BaseProcessor):

    def __init__(
        self,
        input_files: Optional[str] = None,
        preprocessed_dataset_files: Optional[str] = None,
        input_key="question",
        output_key="generation",
        skip_first: int = 0,
        add_correct: bool = True,
        add_incorrect: bool = False,
        use_judgement: bool = False,
        keys_to_keep: list[str] | None = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_files = input_files
        self.preprocessed_dataset_files = preprocessed_dataset_files
        self.input_key = input_key
        self.output_key = output_key
        self.skip_first = skip_first
        self.add_correct = add_correct
        self.add_incorrect = add_incorrect
        self.use_judgement = use_judgement
        self.keys_to_keep = keys_to_keep
        self.deduplicate = deduplicate

        if self.keys_to_keep is not None:
            self.keys_to_keep = set(self.keys_to_keep)
            self.keys_to_keep.add(self.input_key)
            self.keys_to_keep.add(self.output_key)
            self.keys_to_keep.add("is_correct")
            self.keys_to_keep.add("judgement")

        if isinstance(self.input_files, str):
            self.input_files = self.input_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")

        if self.input_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `input_files` or `preprocessed_dataset_files` should be provided")

        if not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

    def _read_preprocessed_data(self, file_handle) -> int:
        samples = []
        questions = set()
        for idx, line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            sample = json.loads(line)
            if self.keys_to_keep:
                sample = {k: v for k, v in sample.items() if k in self.keys_to_keep}
            questions.add(sample[self.input_key])
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
            if not self.use_judgement:
                if "is_correct" not in line_dict:
                    LOG.warning("Found incomplete generations (is_correct field is missing) - skipping")
                    continue

                if not self.add_correct and line_dict["is_correct"]:
                    continue

                if not self.add_incorrect and not line_dict["is_correct"]:
                    continue
            else:
                if "judgement" not in line_dict:
                    LOG.warning("Found incomplete generations (judgement field is missing) - skipping")
                    continue

                if not self.add_correct and is_correct_judgement(line_dict["judgement"]):
                    continue

                if not self.add_incorrect and not is_correct_judgement(line_dict["judgement"]):
                    continue

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

    def _batch_deduplicate(self, batch):
        seen_predictions = defaultdict(set)
        unique_samples = []

        for sample in batch:
            question = sample[self.input_key]
            if sample[self.output_key] not in seen_predictions[question]:
                seen_predictions[question].add(sample[self.output_key])
                unique_samples.append(sample)

        return unique_samples

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def process(self):
        samples = []
        if self.input_files:
            args = [(file, self._read_raw_data) for file in unroll_files(self.input_files)]
            results = process_map(self._parallel_read_file, args, max_workers=4, chunksize=1)
            samples.extend(list(chain(*results)))
        if self.preprocessed_dataset_files:
            args = [(file, self._read_preprocessed_data) for file in unroll_files(self.preprocessed_dataset_files)]
            results = process_map(self._parallel_read_file, args, max_workers=None, chunksize=1)
            samples.extend(list(chain(*results)))

        if self.deduplicate:
            LOG.info("Total samples before deduplication: %d", len(samples))

            # Parallel deduplication
            chunk_size = 100000
            num_cores = max(100, len(samples) // chunk_size)

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Process chunks in parallel
                futures = [
                    executor.submit(self._batch_deduplicate, chunk) for chunk in self._chunks(samples, chunk_size)
                ]

                # Final deduplication of results from all chunks
                seen_predictions = defaultdict(set)
                samples_count = 0

                with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                    for future in futures:
                        chunk_results = future.result()
                        for sample in chunk_results:
                            question = sample[self.input_key]
                            if sample[self.output_key] not in seen_predictions[question]:
                                seen_predictions[question].add(sample[self.output_key])
                                fout.write(json.dumps(sample) + "\n")
                                samples_count += 1

            LOG.info("Total samples after deduplication: %d", samples_count)
        else:
            LOG.info("Total samples: %d", len(samples))
            with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                for sample in samples:
                    fout.write(json.dumps(sample) + "\n")


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
        prompt_config: str,
        prompt_template: str,
        chat_format: str | None = None,  # nemotron/llama/None
        input_key: str = "input",
        output_key: str = "output",
        metadata: Optional[Dict] = None,
        exclude_optional_keys: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.chat_format = chat_format
        self.metadata = metadata
        self.exclude_optional_keys = exclude_optional_keys
        if not self.metadata:
            self.metadata = {}

        self.prompt = None
        if prompt_config and prompt_template:
            self.prompt = get_prompt(prompt_config, prompt_template)
        else:
            LOG.warning("Prompt details are missing! The processed data won't be formatted using any prompt.")

    def process(self):
        samples_count = 0
        seen_predictions = defaultdict(set)
        with (
            open(self.input_manifest_file, "rt", encoding="utf-8") as fin,
            open(self.output_manifest_file, "wt", encoding="utf-8") as fout,
        ):
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                question = elem[self.input_key]
                # deduplication
                if elem[self.output_key] in seen_predictions[question]:
                    continue
                seen_predictions[question].add(elem[self.output_key])
                if 'expected_answer' in elem:
                    elem['expected_answer'] = str(elem['expected_answer'])
                # take only required keys from the input if exclude_optional_keys is True
                output_sample = {}
                if not self.exclude_optional_keys:
                    output_sample = json.loads(line)
                elif "expected_answer" in elem:
                    output_sample["expected_answer"] = elem["expected_answer"]

                if self.chat_format is None:
                    generation = elem.pop(self.output_key)
                    if self.prompt:
                        output_sample["input"] = self.prompt.fill(input_dict=elem)
                        output_sample["output"] = generation + self.prompt.config.template.assistant_end
                    else:
                        output_sample["input"] = elem[self.input_key]
                        output_sample["output"] = generation

                elif self.chat_format.lower() == "nemotron":
                    output_sample['conversations'] = [
                        {
                            'value': self.prompt.config.user.format(**elem) if self.prompt else elem[self.input_key],
                            'from': 'User',
                            'canonical_form': '',
                        },
                        {'value': elem.pop(self.output_key), 'from': 'Assistant', 'canonical_form': ''},
                    ]
                    output_sample['system'] = self.prompt.config.system if self.prompt else ''
                    output_sample['mask'] = 'User'
                elif self.chat_format.lower() == "llama":
                    output_sample['conversations'] = [
                        {
                            'value': self.prompt.config.user.format(**elem) if self.prompt else elem[self.input_key],
                            'from': '<|start_header_id|>user<|end_header_id|>',
                            'canonical_form': '',
                        },
                        {
                            'value': elem.pop(self.output_key),
                            'from': '<|start_header_id|>assistant<|end_header_id|>',
                            'canonical_form': '',
                        },
                    ]
                    output_sample['system'] = self.prompt.config.system if self.prompt else ''
                    output_sample['mask'] = '<|start_header_id|>user<|end_header_id|>'
                else:
                    raise ValueError(f"Chat format {self.chat_format} is not supported")
                output_sample.update(self.metadata)
                fout.write(json.dumps(output_sample) + "\n")
                samples_count += 1

        LOG.info("Prepared dataset size: %d", samples_count)
