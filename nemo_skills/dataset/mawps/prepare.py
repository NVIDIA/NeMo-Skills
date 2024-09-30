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

# We're downloading this dataset from the ToRA repo because
# the official download link is currently down.
URL = "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/{subset}.jsonl"


subsets = ["addsub", "singleeq", "singleop", "multiarith"]

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"test.jsonl")

    data = []
    for subset in subsets:
        original_file = str(data_dir / f"original_{subset}.jsonl")

        if not os.path.exists(original_file):
            urllib.request.urlretrieve(URL.format(subset=subset), original_file)

        with open(original_file, "r") as fin:
            for line in fin:
                original_entry = json.loads(line)
                new_entry = dict(
                    problem=original_entry["input"],
                    expected_answer=original_entry["target"],
                    type=subset,
                )
                # converting to int if able to for cleaner text representation
                if int(new_entry["expected_answer"]) == new_entry["expected_answer"]:
                    new_entry["expected_answer"] = int(new_entry["expected_answer"])

                data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
