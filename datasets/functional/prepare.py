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
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/ConsequentAI/fneval/raw/main/{}.tar.gz"
# Data Format
#
# Required:
#   - question (problem statement)
#
# Optional:
#   - expected_answer (expected answer)
#   - reference_solution (text-based solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default='Dec-2023',
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "Algebra",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Number Theory",
            "Prealgebra",
            "Precalculus",
        ],
    )
    args = parser.parse_args()

    data_folder = Path(__file__).absolute().parent
    data_folder.mkdir(exist_ok=True)
    original_file = str(data_folder / f"original_test_{args.date}")
    output_file = str(data_folder / "test.jsonl")

    archive_path = original_file + '.tar.gz'
    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(URL.format(args.date), archive_path)
        # Open and extract the tar.gz file
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(original_file)

    fin_data = []
    for category in args.categories:
        category_id = category.replace('&', 'and').replace(' ', '_').lower()
        path = os.path.join(original_file, args.date, 'test', category_id)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                data = json.load(f)
            fin_data.append(
                {
                    'question': data['problem'],
                    'expected_answer': data['solution'][7:-1],
                    'id': f'test/{category_id}/{file[:-5]}.json',
                    'date': args.date,
                    'category': category,
                }
            )

    with open(output_file, "wt", encoding="utf-8") as fout_ic:
        for original_entry in fin_data:
            # original entries
            if original_entry["category"] in args.categories and args.date == original_entry['date']:
                entry = dict(
                    question=original_entry["question"],
                    expected_answer=original_entry["expected_answer"],
                    **{
                        key: value
                        for key, value in original_entry.items()
                        if key not in ["expected_answer", "question"]
                    },
                )
                fout_ic.write(json.dumps(entry) + "\n")
