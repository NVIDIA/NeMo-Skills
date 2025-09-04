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
from pathlib import Path
from typing import List

import pandas
from tqdm import tqdm

# SimpleQA dataset from OpenAI's simple-evals repository


def format_entry(entry: dict, idx: int) -> dict:
    """Format an entry to match NeMo-Skills format."""
    return {
        "id": entry.get("id", f"simpleqa_{idx}"),
        "metadata": eval(entry["metadata"]),
        "problem": entry["problem"],
        "expected_answer": entry["answer"],
    }


def write_data_to_file(output_file, examples: List[dict]):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for row in tqdm(examples, desc=f"Writing {output_file.name}"):
            json.dump(row, fout)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test"),
        help="Dataset split to process; only one split.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    # Download the SimpleQA dataset

    output_file = data_dir / f"{args.split}.jsonl"

    df = pandas.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")

    # Columns: metadata,problem,answer
    # "{'topic': 'Science and technology', 'answer_type': 'Person', 'urls': ['https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award', 'https://ieeexplore.ieee.org/author/37271220500', 'https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award', 'https://www.nxtbook.com/nxtbooks/ieee/awards_2010/index.php?startid=21#/p/20']}",Who received the IEEE Frank Rosenblatt Award in 2010?,Michio Sugeno

    # format the dataset
    examples = []
    for idx, row in df.iterrows():
        formatted_entry = format_entry(row, idx)
        examples.append(formatted_entry)

    # convert to jsonl
    write_data_to_file(output_file, examples)
