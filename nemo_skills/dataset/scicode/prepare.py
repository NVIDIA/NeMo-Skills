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

URL = "https://huggingface.co/datasets/SciCode1/SciCode/raw/main/problems_{split}.jsonl"


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    for split in ["dev", "test"]:
        original_file = str(data_dir / f"original_{split}.jsonl")
        data_dir.mkdir(exist_ok=True)
        output_file = str(data_dir / f"{split}.jsonl")

        if not os.path.exists(original_file):
            urllib.request.urlretrieve(URL.format(split=split), original_file)

        data = []
        with open(original_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                entry = json.loads(line)
                new_entry = entry  # TODO?
                data.append(new_entry)

        with open(output_file, "wt", encoding="utf-8") as fout:
            for entry in data:
                fout.write(json.dumps(entry) + "\n")
