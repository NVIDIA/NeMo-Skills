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
import copy
import json
import os

from tqdm import tqdm
from datasets import load_dataset

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}


def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Open Code Reasoning questions")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared questions")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the OCR2 dataset and prepare questions
    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    # Save only subset that contains unique question_ids
    unique_values = set()  # To keep track of unique values encountered
    first_occurrence_indices = []

    for ocr2_ds in [ocr2_dataset["python"]]:
        for ocr2_ds_item in tqdm(ocr2_ds):
            assert ocr2_ds_item["dataset"] in ["taco", "apps", "code_contests", "open-r1/codeforces"]
            ds_name, ds_split, ds_index = ocr2_ds_item["dataset"], ocr2_ds_item["split"], int(ocr2_ds_item["index"])
            question = get_question(ds_name, ds_split, ds_index)
            assert question is not None
            assert ocr2_ds_item["question"] == "-"
            # Update the question field with the retrieved question
            ocr2_ds_item["question"] = question

            # Delete the solution data
            ocr2_ds_item["r1_generation"] = ""
            ocr2_ds_item["qwq_critique"] = ""
            ocr2_ds_item["solution"] = ""

            if ocr2_ds_item["question_id"] not in unique_values:
                unique_values.add(ocr2_ds_item["question_id"])
                first_occurrence_indices.append(copy.deepcopy(ocr2_ds_item))

    # Save the updated dataset
    output_filepath = os.path.join(output_dir, "open_code_reasoning_questions.jsonl")
    with open(output_filepath, "w") as f:
        for item in first_occurrence_indices:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared questions saved to {output_filepath}")


