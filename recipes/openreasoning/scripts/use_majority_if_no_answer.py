# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import json
import os
from collections import Counter
from typing import Union

from nemo_skills.evaluation.metrics.utils import is_correct_judgement


def process_files(input_folder: str, output_folder: str) -> None:
    # Find all output-rsX.jsonl files
    file_paths = glob.glob(os.path.join(input_folder, "output-rs*.jsonl"))

    if not file_paths:
        print(f"No output-rs*.jsonl files found in {input_folder}")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load all files
    all_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            all_data.append((file_path, data))

    # Check if all files have the same number of lines
    line_counts = [len(data) for _, data in all_data]
    if len(set(line_counts)) > 1:
        print(f"Warning: Files have different line counts: {line_counts}")

    num_files = len(all_data)
    num_lines = max(line_counts) if line_counts else 0

    changes_made = 0
    incorrect_judgements = 0

    # Process each line across all files
    for line_idx in range(num_lines):
        # Check if any file has a correct judgement for this line
        any_correct_judgement = False
        predicted_answers = []

        for _, data in all_data:
            line_data = data[line_idx]
            judgement = line_data["judgement"]

            if is_correct_judgement(judgement):
                any_correct_judgement = True
                break

            predicted_answer = line_data["predicted_answer"]
            if predicted_answer is not None:
                predicted_answers.append(predicted_answer)

        # If no correct judgement found, update predicted_answer with majority predicted_answer
        if not any_correct_judgement:
            if not predicted_answers:
                # If no predicted answers, skip this line
                print(f"No predicted answers for line {line_idx} in any file, skipping")
                continue
            incorrect_judgements += 1

            # Find majority predicted_answer
            majority_answer = Counter(predicted_answers).most_common(1)[0][0]

            # Update the predicted_answer in each file for this line
            for _, data in all_data:
                if data[line_idx]["expected_answer"] != majority_answer:
                    data[line_idx]["updated_answer_from_r1"] = True
                data[line_idx]["expected_answer"] = majority_answer
                changes_made += 1

    # Save modified files
    for file_path, data in all_data:
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        with open(output_file, 'w') as f:
            for line_data in data:
                f.write(json.dumps(line_data) + '\n')

    print(f"Processed {num_files} files with {num_lines} lines")
    print(f"Found {incorrect_judgements} lines without any correct judgement")
    print(f"Made {changes_made} changes to expected_answer values")


def main():
    parser = argparse.ArgumentParser(description='Process JSONL files and update predicted answers')
    parser.add_argument('input_folder', help='Folder containing output-rsX.jsonl files')
    parser.add_argument('output_folder', help='Folder to save modified files')
    args = parser.parse_args()

    process_files(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
