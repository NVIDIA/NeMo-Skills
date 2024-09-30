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

from nemo_skills.dataset.utils import add_rounding_instruction

URL = "https://huggingface.co/datasets/qintongli/GSM-Plus/resolve/main/data/test-00000-of-00001.jsonl?download=true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "adding_operation",
            "critical_thinking",
            "digit_expansion",
            "distractor_insertion",
            "integer-decimal-fraction_conversion",
            "numerical_substitution",
            "problem_understanding",
            "reversing_operation",
        ],
    )
    parser.add_argument("--no_rounding_instructions", action='store_true')
    args = parser.parse_args()

    split = "test"
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    original_file = str(data_dir / f"original_{split}.jsonl")
    output_file = str(data_dir / f"{split}.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)

    file_rounded = None
    if not args.no_rounding_instructions:
        output_file_rounded = str(data_dir / f"{split}_rounded.jsonl")
        file_rounded = open(output_file_rounded, 'w')

    with open(original_file, "rt") as original, open(output_file, "w") as test_full:
        original_data = [json.loads(line) for line in original.readlines()]
        for original_entry in original_data:
            if original_entry["perturbation_type"].replace(' ', '_') in args.categories:
                # original entries
                reference_solution = original_entry.get("solution", None) or original_entry.get(
                    "reference_solution", None
                )
                expected_answer = original_entry.get("answer", None) or original_entry.get("expected_answer", None)
                entry = dict(
                    problem=original_entry["question"],
                    reference_solution=reference_solution,
                    expected_answer=expected_answer,
                    **{
                        key: value
                        for key, value in original_entry.items()
                        if key
                        not in [
                            "answer",
                            "expected_answer",
                            "solution",
                            "problem",
                            "reference_solution",
                        ]
                    },
                )
                # converting to int if able to for cleaner text representation
                if str(entry["expected_answer"]).replace('.', "", 1).replace('-', "", 1).isdigit():
                    entry["expected_answer"] = float(entry["expected_answer"])
                    if int(entry["expected_answer"]) == entry["expected_answer"]:
                        entry["expected_answer"] = int(entry["expected_answer"])

                test_full.write(json.dumps(entry) + "\n")

                if file_rounded:
                    entry_rounded = add_rounding_instruction(entry)
                    file_rounded.write(json.dumps(entry_rounded) + "\n")

    if file_rounded:
        file_rounded.close()
