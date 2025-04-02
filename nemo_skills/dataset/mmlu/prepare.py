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
import csv
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path

URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

# mmlu subcategories from https://github.com/hendrycks/test/blob/master/categories.py
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}


def read_csv_files_from_tar(tar_file_path, split):
    result = {}

    # Define the column names
    column_names = ["question", "A", "B", "C", "D", "expected_answer"]

    with tarfile.open(tar_file_path, 'r') as tar:
        # List all members of the tar file
        members = tar.getmembers()

        # Filter for CSV files in the 'data/test' directory
        csv_files = [
            member for member in members if member.name.startswith(f'data/{split}/') and member.name.endswith('.csv')
        ]

        for csv_file in csv_files:
            # Extract the file name without the path
            file_name = os.path.basename(csv_file.name)

            # Read the CSV file content
            file_content = tar.extractfile(csv_file)
            if file_content is not None:
                # Decode bytes to string
                content_str = io.TextIOWrapper(file_content, encoding='utf-8')

                # Use csv to read the CSV content without a header
                csv_reader = csv.reader(content_str)

                # Convert CSV data to list of dictionaries with specified column names
                csv_data = []
                for row in csv_reader:
                    if len(row) == len(column_names):
                        csv_data.append(dict(zip(column_names, row)))
                    else:
                        print(f"Warning: Skipping row in {file_name} due to incorrect number of columns")

                # Add to result dictionary
                result[file_name.rsplit('_', 1)[0]] = csv_data

    return result


def save_data(split):
    data_dir = Path(__file__).absolute().parent
    data_file = str(data_dir / f"data.tar")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"{split}.jsonl")

    urllib.request.urlretrieve(URL, data_file)

    original_data = read_csv_files_from_tar(data_file, split)
    data = []
    for subject, questions in original_data.items():
        for question in questions:
            new_entry = question
            new_entry['subtopic'] = subject
            new_entry['problem'] = new_entry.pop('question')
            new_entry['options'] = "\n".join(f"{chr(ord('A') + i)}. {new_entry[chr(ord('A') + i)]}" for i in range(4))
            new_entry['problem'] += '\n' + new_entry['options']

            new_entry['subset_for_metrics'] = subcategories[subject][0]
            data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")

    # cleaning up the data file to avoid accidental upload on clusters
    os.remove(data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("dev", "test", "val"),
    )
    args = parser.parse_args()

    if args.split == "all":
        for split in ["dev", "test", "val"]:
            save_data(split)
    else:
        save_data(args.split)
