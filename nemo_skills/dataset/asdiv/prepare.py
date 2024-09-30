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
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    original_file = str(data_dir / f"original_test.xml")
    output_file = str(data_dir / f"test.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    tree = ET.parse(original_file)
    root = tree.getroot()

    with open(output_file, "wt", encoding="utf-8") as fout:
        for key, problem in enumerate(root.iter("Problem")):
            new_entry = dict(
                problem=problem.find("Body").text.strip() + ' ' + problem.find("Question").text.strip(),
                expected_answer=problem.find("Answer").text.split('(')[0].strip(),
                type=problem.find("Solution-Type").text,
            )
            # converting to int if able to for cleaner text representation
            try:
                if int(new_entry["expected_answer"]) == new_entry["expected_answer"]:
                    new_entry["expected_answer"] = int(new_entry["expected_answer"])
            except:
                pass

            fout.write(json.dumps(new_entry) + "\n")
