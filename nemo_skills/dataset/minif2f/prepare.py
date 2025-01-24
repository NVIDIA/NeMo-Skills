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

import argparse
import json
import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/main/datasets/minif2f.jsonl"


def download_dataset(output_path):
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(URL, output_path)


def split_data(input_file):
    valid_data = []
    test_data = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if entry['split'] == 'valid':
                valid_data.append(entry)
            elif entry['split'] == 'test':
                test_data.append(entry)

    return valid_data, test_data


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def main(split):
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "minif2f.jsonl")
    valid_file = str(data_dir / "valid.jsonl")
    test_file = str(data_dir / "test.jsonl")

    download_dataset(original_file)
    valid_data, test_data = split_data(original_file)

    if split == "valid":
        save_data(valid_data, valid_file)
    elif split == "test":
        save_data(test_data, test_file)
    elif split == "all":
        save_data(valid_data, valid_file)
        save_data(test_data, test_file)

    delete_file(original_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=("all", "test", "valid"), help="Data split to process")
    args = parser.parse_args()

    main(args.split)
