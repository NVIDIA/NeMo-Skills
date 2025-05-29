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
import random
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields


domains = ["AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
           "Electrodynamics", "GeometricalOptics", "QuantumMechanics",
           "Relativity", "SemiconductorPhysics", "Solid-StatePhysics", "StatisticalMechanics",
           "TheoreticalMechanics", "Thermodynamics", "WaveOptics"]

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def format_entry(entry):
    return {
        "expected_answer": entry["answers"],
        "subset_for_metrics": entry["subject"],
        "domain": entry["domain"],
        "topic": entry["topic"],
        "is_multiple_answer": entry["is_multiple_answer"],
        "level": entry["level"],
        "answer_type": entry["answer_type"],
        "unit": entry["unit"],
        "index": entry["index"],
        "solution":entry["solution"],
        "language": entry["language"]
    }


def save_data(language: str):
    num_multiple_answers = 0
    num_samples = 0
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"test_{language}.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for domain in domains:
            dataset = load_dataset("UGPhysics/ugphysics", domain, split=language)
            for entry in dataset:
                num_samples += 1
                if entry["is_multiple_answer"]:
                    num_multiple_answers += 1
                json.dump(format_entry(entry), fout)
                fout.write("\n")
    print(f"Saved {num_samples} entries to {output_file}")
    print(f"Number of multiple choice questions: {num_multiple_answers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        default="en",
        choices=("en", "zh"),
        help="Language of the dataset.",
    )
    args = parser.parse_args()

    if args.language == "all" and False:
        for language in ["en", "zh"]:
            save_data(language)
    else:
        save_data(args.language)
