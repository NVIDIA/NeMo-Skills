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
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from utils import prepare_for_sft

# utils is adding main package to path already
from nemo_skills.inference.prompt.utils import prompt_types

URL = "https://huggingface.co/datasets/nvidia/OpenMath-MATH-masked/resolve/main/data.jsonl?download=true"


# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_name",
        required=True,
        choices=("validation", "train", "train_full"),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--prompt_type", default="openmathinstruct/sft", choices=prompt_types)
    args = parser.parse_args()

    data_folder = Path(__file__).absolute().parent
    original_file = str(data_folder / f"original_train.jsonl")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / f"{args.split_name}.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(
            URL,
            original_file,
        )

    with open(original_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    random.seed(args.random_seed)
    random.shuffle(data)
    if args.split_name == "validation":
        data = data[: args.validation_size]
        # dumping SFT-ready validation file as well right away
        with open(data_folder / "validation-sft.jsonl", "wt", encoding="utf-8") as fout:
            for entry in prepare_for_sft(data, args.prompt_type, "math-masked", chat_format=False):
                fout.write(json.dumps(entry) + "\n")
        with open(data_folder / "validation-sft-chat.jsonl", "wt", encoding="utf-8") as fout:
            for entry in prepare_for_sft(data, args.prompt_type, "math-masked", chat_format=True):
                fout.write(json.dumps(entry) + "\n")
    elif args.split_name == "train":
        data = data[args.validation_size :]

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
