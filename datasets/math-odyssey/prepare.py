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

URL = "https://raw.githubusercontent.com/protagolabs/odyssey-math/main/final-odyssey-math-with-levels.jsonl"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


def identify_label(answer_endings, answer):
    for ending in answer_endings:
        if answer.endswith(ending):
            answer = answer[: -(len(ending))]
            break
    return answer


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
    answer_endings = [
        "\\\\\n\\noindent",
        "\\\\\n\n\\noindent",
        "\\\\\n\t\\noindent",
        ".\n\n\\noindent",
        "\n\n\\noindent",
        "\\\\\n\n  \n\t\\noindent",
        "\\\\ \n\t\\noindent",
        "\\\\\n\n\t\\noindent",
    ]
    with open(original_file, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            new_entry = {}

            original_entry = json.loads(line)  # Convert JSON line to dictionary
            key = list(original_entry.keys())[0]
            original_entry = original_entry[key]
            # mapping to the required naming format
            new_entry["question"] = original_entry["question"]
            answer = original_entry["answer"]
            for ending in answer_endings:
                if answer.endswith(ending):
                    ### remove all white space and remove all $ sign so that we can match previous formats
                    answer = answer[: -(len(ending))].strip()
                    if answer.startswith("\\") or answer.endswith("\\"):
                        answer = answer.strip('\\').strip()
                    if answer[-1] == '.':
                        answer = answer[:-1]
                    if "$" in answer:
                        answer = answer.replace('$', '').strip()

            new_entry["expected_answer"] = answer
            new_entry["original_answer"] = original_entry["answer"]
            new_entry["reference_solution"] = original_entry["reasoning"]
            new_entry["label"] = original_entry["label"]
            new_entry["level"] = original_entry["level"]

            data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
