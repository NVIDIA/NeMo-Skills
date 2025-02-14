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
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


# mmlu subsets from https://github.com/hendrycks/test/blob/master/categories.py
all_categories = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology',
                  'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
                  'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
                  'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
                  'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
                  'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                  'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
                  'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
                  'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
                  'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']


def format_entry(entry, category):
    return {
        "question": entry['question'],
        "A": entry['choices'][0],
        "B": entry['choices'][1],
        "C": entry['choices'][2],
        "D": entry['choices'][3],
        "expected_answer": chr(65 + entry['answer']),
        "expected_answer_corrected": entry['correct_answer'],
        "error_type": entry['error_type'],
        "category": category,
        "potential_reason": entry['potential_reason']
    }


def write_data_to_file(output_file, data, category):
    with open(output_file, "at", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {category} to {output_file.name}"):
            json.dump(format_entry(entry, category), fout)
            fout.write("\n")


def main(args):
    # Create the output directory if it doesn't exist
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    print(f"Loading categories: {all_categories}")

    # create output_file or remove its contents if it exists
    output_file = data_dir / f"{args.split}.jsonl"
    open(output_file, "w")

    # Load the dataset and write it to the output 
    for category in tqdm(all_categories):
        dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", name=category, split='test')
        write_data_to_file(output_file, dataset, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all", choices=(all_categories), help="Dataset category subset to process.")
    parser.add_argument("--split", default="test", choices=(["test"]), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
