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
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / f"input_data.jsonl")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"test.jsonl")

    urllib.request.urlretrieve(URL, original_file)

    with open(original_file, "rt", encoding="utf-8") as fin:
        original_data = [json.loads(line) for line in fin]

    data = []
    for original_entry in original_data:
        new_entry = original_entry
        new_entry["question"] = original_entry["prompt"]
        data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
