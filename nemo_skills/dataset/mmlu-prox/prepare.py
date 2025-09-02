# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from lang_libs import LANG_LIBS, LANG_SUBJECTS
from tqdm import tqdm


def format_entry(entry, language):
    category = entry["category"].replace(" ", "_")  # Fix computer science category

    entry["options"] = []
    for i in range(10):
        entry["options"].append(entry[f"option_{i}"])
        del entry[f"option_{i}"]

    subject = LANG_SUBJECTS[language][category]
    description = LANG_LIBS[language][3].format(subject=subject, ans_suffix=LANG_LIBS[language][5].format("X")) + "\n"

    def get_mcq_fields(question, choices, language):
        options_dict = {chr(ord("A") + i): option for i, option in enumerate(choices)}
        options_text = "\n".join(f"{letter}. {option}" for letter, option in options_dict.items())
        return {
            "question": f"{description}{LANG_LIBS[language][0]}\n{question}\n{LANG_LIBS[language][1]}\n{options_text}\n",
            "options": options_text,
            **options_dict,
        }

    extract_regex = LANG_LIBS[language][5].replace("({})", r"\(?([ABCDEFGHIJ])\)?")
    if language == "en":
        extract_regex = extract_regex.lstrip("the").strip()
        extract_regex = extract_regex.replace("\\(", "\\**\\(")
        extract_regex = extract_regex.replace("\\)?", "\\)?\\**")

    return {
        "expected_answer": entry["answer"],
        "extract_from_boxed": "False",
        "extract_regex": extract_regex,
        "subset_for_metrics": language,
        "category": category,
        **get_mcq_fields(entry["question"], entry["options"], language),
    }


def write_data_to_file(output_file, datasets, languages):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, dataset in enumerate(datasets):
            for entry in tqdm(dataset, desc=f"Writing {output_file.name}"):
                json.dump(format_entry(entry, language=languages[idx]), fout)
                fout.write("\n")


def main(args):
    datasets = [load_dataset("li-lab/MMLU-ProX", lang)[args.split] for lang in args.languages]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, datasets, languages=args.languages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("validation", "test"), help="Dataset split to process.")
    parser.add_argument(
        "--languages", default=["en", "de", "es", "fr", "it", "ja"], nargs="+", help="Languages to process."
    )
    args = parser.parse_args()
    main(args)
