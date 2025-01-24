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

LABEL = "label"
URL = "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_mstep.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_label",
        choices=[
            "in_range",
            "out_range",
        ],
    )
    parser.add_argument(
        "--role_label",
        choices=[
            "overlapped",
            "nonoverlapped",
        ],
    )
    parser.add_argument(
        "--sentence_label",
        choices=[
            "in_topic",
            "out_topic",
        ],
    )
    args = vars(parser.parse_args())
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    original_file = str(data_dir / "original_test.json")
    output_file_ic = str(data_dir / "test.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    with open(original_file, "rt") as fin, open(output_file_ic, "wt", encoding="utf-8") as fout_ic:
        fin_data = json.loads(fin.read())
        for original_entry in fin_data:
            # entries with irrelevant context
            if all(
                [original_entry[key] == value for key, value in args.items() if LABEL in key and value is not None]
            ):
                ic_entry = dict(
                    problem=original_entry["new_question"],
                    expected_answer=original_entry["answer"],
                    **{key: value for key, value in original_entry.items() if key not in ["answer", "new_question"]},
                )
                # converting to int if able to for cleaner text representation
                if str(ic_entry["expected_answer"]).replace('.', "", 1).replace('-', "", 1).isdigit():
                    ic_entry["expected_answer"] = float(ic_entry["expected_answer"])
                    if int(ic_entry["expected_answer"]) == ic_entry["expected_answer"]:
                        ic_entry["expected_answer"] = int(ic_entry["expected_answer"])
                fout_ic.write(json.dumps(ic_entry) + "\n")
