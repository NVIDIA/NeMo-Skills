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
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/AA-LCR_Dataset.csv"


def save_data(split):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    
    original_file = str(data_dir / "original_data.csv")
    output_file = str(data_dir / f"{split}.jsonl")
    
    # Download the dataset
    urllib.request.urlretrieve(URL, original_file)
    
    # Load and process the data
    df = pd.read_csv(original_file)
    
    data = []
    for _, row in df.iterrows():
        new_entry = {
            "problem": row["question"],
            "expected_answer": row["answer"],
            "document_category": row["document_category"],
            "document_set_id": row["document_set_id"],
            "question_id": row["question_id"],
            "input_tokens": row["input_tokens"],
            "data_source_filenames": row["data_source_filenames"],
            "data_source_urls": row["data_source_urls"]
        }
        data.append(new_entry)
    
    # Write to JSONL format
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
    
    # Clean up original file
    import os
    os.remove(original_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="AA-LCR is a test-only dataset"
    )
    args = parser.parse_args()
    
    save_data(args.split)