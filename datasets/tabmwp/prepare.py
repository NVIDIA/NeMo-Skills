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

URL_prefix = "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_"

# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Reads TabMWP data and converts it to a format readable by the llm-structured-data scripts."
    )
    parser.add_argument(
        "--data_dir",
        default=f"{Path(__file__).absolute().parent / 'original'}",
        help="Path to directory with JSON files with TabMWP data. Will automatically download if not found.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{Path(__file__).absolute().parent}",
        help="Path to where the output file will be written.",
    )
    args = parser.parse_args()

    splits = ["train", "dev", "test"]

    for split in splits:
        original_file = os.path.join(args.data_dir, f"problems_{split}.json")

        if not os.path.exists(original_file):
            os.makedirs(args.data_dir, exist_ok=True)
            urllib.request.urlretrieve(URL_prefix + split + ".json", original_file)

        output_file = (
            os.path.join(args.output_dir, "validation.jsonl")
            if split == "dev"
            else os.path.join(args.output_dir, split + ".jsonl")
        )

        question_template = (
            "Read the following table and then answer the question that follows.\n{table}\n\n{question}"
        )

        with open(original_file, "r") as fin:
            data = json.load(fin)

        with open(output_file, "wt", encoding="utf-8") as fout:
            for original_entry in data.values():
                question = question_template.format(
                    table_title=original_entry["table_title"],
                    table=original_entry["table"],
                    question=original_entry["question"],
                )

                new_entry = dict(
                    question=question,
                    expected_answer=original_entry["answer"],
                    reference_solution=original_entry["solution"],
                    table_title=original_entry["table_title"],
                    table=original_entry["table_for_pd"],
                    grade=original_entry["grade"],
                )

                if original_entry["ques_type"] == "multi_choice":
                    new_entry["question"] += "\nAnswer options:\n{}".format(", ".join(original_entry["choices"]))

                # converting to int if able to for cleaner text representation
                if original_entry["ans_type"] == "integer_number":
                    new_entry["expected_answer"] = int(new_entry["expected_answer"].replace(",", ""))

                fout.write(json.dumps(new_entry) + "\n")
