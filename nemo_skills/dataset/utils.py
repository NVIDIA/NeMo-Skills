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

import contextlib
import importlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict
from urllib.error import URLError
from nemo_skills.dataset.parser_qwen import parse_ground_truth

@contextlib.contextmanager
def add_to_path(p):
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path = old_path


def add_rounding_instruction(data: Dict) -> Dict:
    try:
        float(data['expected_answer'])
        number_of_values = 0
        if '.' in str(data['expected_answer']):
            number_of_values = len(str(data['expected_answer']).split('.')[1])
        if number_of_values == 0:
            data['problem'] += ' Express the answer as an integer.'
        elif number_of_values == 1:
            data['problem'] += ' Round the answer to one decimal place.'
        else:
            data['problem'] += f' Round the answer to {number_of_values} decimal places.'
    except ValueError:
        pass
    return data


def get_dataset_module(dataset, extra_datasets=None):
    try:
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        found_in_extra = False
    except ModuleNotFoundError:
        extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
        if extra_datasets is None:
            raise
        with add_to_path(extra_datasets):
            dataset_module = importlib.import_module(dataset)
        found_in_extra = True
    return dataset_module, found_in_extra


def download_with_retries(url, output_file, max_retries=3, retry_delay=1):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_file)
            return True
        except URLError as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts: {e}")
            time.sleep(retry_delay * (attempt + 1))
    return False


def save_data_from_qwen(dataset, split="test"):
    url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/evaluation/data/{dataset}/{split}.jsonl"
    )

    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / dataset / f"original_{split}.json")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / dataset / f"{split}.jsonl")
    data = []
    if not os.path.exists(original_file):
        formatted_url = url.format(split=split, dataset=dataset)
        download_with_retries(formatted_url, original_file)

    with open(original_file, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            entry = json.loads(line)
            # TODO: add else

            if "problem" not in entry:
                entry["problem"] = entry.pop("question")
            # use parse_ground_truth from qwen repo to get the expected answer
            _, expected_answer = parse_ground_truth(entry, dataset)
            entry["expected_answer"] = expected_answer
                
            data.append(entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


    return output_file