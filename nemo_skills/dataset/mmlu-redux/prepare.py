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
    answer, correct_answer, error_type = entry['answer'], entry['correct_answer'], entry['error_type']
    if entry["error_type"] in ["bad_question_clarity", "bad_options_clarity", "expert"]:
        return None
    if error_type in ["wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"] and not correct_answer:
        return None
    if error_type == "ok":
        assert chr(65 + answer) in "ABCD", f"Error in Ok question: {answer}"
        final_answer = chr(65 + answer)
    elif error_type == "wrong_groundtruth" and correct_answer:
        if correct_answer in "ABCD":
            final_answer = correct_answer
        else:
            assert chr(65 + int(correct_answer)) in "ABCD", f"Error in Wrong Groundtruth question: {answer}"
            final_answer = chr(65 + int(correct_answer))
    elif error_type == "no_correct_answer" and correct_answer:
        if '?' in correct_answer:
            return None
        # if the text in correct_answer is approx the same length as the choices it is the answer
        answers_length = [len(choice) for choice in entry['choices']]
        if -1 <= len(correct_answer) - min(answers_length) and len(correct_answer) - max(answers_length) <= 1: 
            entry['choices'][answer] = correct_answer
            final_answer = chr(65 + int(answer))
        else:
            return None
    elif error_type == "multiple_correct_answers" and correct_answer:
        # keep only ABCD0123
        correct_answer = [letter for letter in correct_answer if letter in 'ABCD0123']
        if correct_answer:
            final_answer = ','.join(ans if ans in 'ABCD' else chr(65 + int(ans)) for ans in correct_answer)
        else:
            return None
    else:
        return None
    return {
        "question": entry['question'],
        "A": entry['choices'][0],
        "B": entry['choices'][1],
        "C": entry['choices'][2],
        "D": entry['choices'][3],
        "expected_answer": final_answer,
        "category": category,
        "source": entry['source']
    }


def write_data_to_file(output_file, data, category):
    with open(output_file, "at", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {category} to {output_file.name}"):
            if (final_entry := format_entry(entry, category)):
                json.dump(final_entry, fout)
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
