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
import subprocess
from pathlib import Path

URL = "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl"


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    original_file = str(data_folder / "original_test.json")
    data_folder.mkdir(exist_ok=True)
    output_file = str(data_folder / "test.jsonl")

    subprocess.run(
        ["wget", "-q", "--no-check-certificate", "--content-disposition", "-O", original_file, URL], check=True
    )

    data = []

    with open(original_file, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            entry = json.loads(line)
            entry["expected_answer"] = entry.pop("answer")
            entry['reference_solution'] = entry.pop('solution')
            data.append(entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")

    os.remove(original_file)
