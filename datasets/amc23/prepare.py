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
import os
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/QwenLM/Qwen2-Math/main/evaluation/data/amc23/test.jsonl"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    original_file = str(data_folder / "original_test.json")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / "test.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    data = []

    #### For this dataset, it contains 387 examples, but the answers have varying ending formats.
    #### I manually checked all the different types and extracted only the answers

    with open(original_file, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):

            entry = json.loads(line)  # Convert JSON line to dictionary

            answer = entry["answer"]
            entry["expected_answer"] = answer

            data.append(entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
