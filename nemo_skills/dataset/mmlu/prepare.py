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
            new_entry['subset_for_metrics'] = subject
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
