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
import random
import re
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from utils import prepare_for_sft

from nemo_skills.code_execution.math_grader import extract_answer, normalize_answer_string

# utils is adding main package to path already
from nemo_skills.inference.prompt.utils import prompt_types

DOWNLOAD_LINK = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"


def extract_attributes_from_name(file_name):
    """Extract attributes from file path."""
    eval_set, problem_type, fileid = file_name.split("/")[1:]
    fileid = fileid.split(".")[0]
    return eval_set, problem_type, fileid


def extract_answer_string_2(answer_str):
    """For two cases, inside the boxed expression, we needed a second iteration of parsing."""
    left_string = "\\boxed"
    idx = answer_str.rfind(left_string)

    stripped_answer = answer_str[idx + len(left_string) :]
    right_idx = stripped_answer.rfind("$")

    stripped_answer = stripped_answer[:right_idx]
    return stripped_answer


def _post_fix(problem_id, soln_string):
    """Post fixing some answer strings"""
    if problem_id == "test/intermediate_algebra/78.json":
        soln_string = re.sub(r"\\(\d+)", r"\1", soln_string)

    if problem_id == "train/number_theory/7115.json":
        return "A"

    if problem_id == "train/number_theory/1012.json":
        return "E"

    if problem_id == "train/prealgebra/666.json":
        return "125"

    if problem_id == "train/intermediate_algebra/172.json":
        return "two lines"

    if problem_id == "train/prealgebra/1691.json":
        return "1.85"

    if problem_id == "train/geometry/6177.json":
        return "C"

    if problem_id == "train/number_theory/7117.json":
        return "A"

    if problem_id == "train/geometry/6202.json":
        return "D"

    if problem_id == "train/precalculus/268.json":
        return "A"

    return soln_string


def process_data():
    """Download tar and condense data into single jsonl file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_name",
        required=True,
        choices=("test", "validation", "train", "train_full"),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--prompt_type", default="openmathinstruct/sft", choices=prompt_types)
    args = parser.parse_args()

    output_folder = Path(__file__).absolute().parent
    output_folder.mkdir(exist_ok=True)
    actual_split_name = "test" if args.split_name == "test" else "train"

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_filename = os.path.join(temp_dir, "temp.tar")
        urllib.request.urlretrieve(DOWNLOAD_LINK, archive_filename)

        split_instances_dict = defaultdict(list)

        with tarfile.TarFile(archive_filename, mode="r") as reader_f:
            for tar_member in reader_f:
                filename = tar_member.name
                if not filename.endswith(".json"):
                    continue

                eval_set, problem_type, fileid = extract_attributes_from_name(filename)
                # TODO: we should just process all at ones, not do duplicate computation
                if eval_set != actual_split_name:
                    continue

                content = json.loads(reader_f.extractfile(tar_member).read())
                content["question"] = content["problem"]
                content["reference_solution"] = content["solution"]
                del content["problem"]
                del content["solution"]

                answer_string = extract_answer(content["reference_solution"])

                if answer_string is None:
                    answer_string = extract_answer_string_2(content["reference_solution"])

                parsed_answer = normalize_answer_string(answer_string)
                if not (
                    ("Find the equation" in content["question"])
                    or ("Enter the equation" in content["question"])
                    or ("What is the equation") in content["question"]
                    or ("described by the equation") in content["question"]
                    or ("Find an equation") in content["question"]
                ) and ("=" in parsed_answer):
                    if parsed_answer.count("=") == 1:
                        # For greater count, it means we're just predicting values of multiple variables
                        parsed_answer = parsed_answer.split("=")[1]
                content["expected_answer"] = parsed_answer

                # Sanity check that content type matches the parent folder
                content_type = content["type"].lower()
                content_type = content_type.replace(" ", "_")
                content_type = content_type.replace("&", "and")
                assert problem_type == content_type

                content["id"] = f"{eval_set}/{problem_type}/{fileid}.json"
                content["expected_answer"] = _post_fix(content["id"], content["expected_answer"])

                split_instances_dict[eval_set].append(content)

        assert len(split_instances_dict) == 1
        for split, instances in split_instances_dict.items():
            # always shuffling to make it easier to get validation/train out of train_full
            if args.split_name != "test":
                random.seed(args.random_seed)
                random.shuffle(instances)
            if args.split_name == "validation":
                data = instances[: args.validation_size]
                # dumping SFT-ready validation file as well right away
                with open(output_folder / "validation-sft.jsonl", "wt", encoding="utf-8") as fout:
                    for entry in prepare_for_sft(data, args.prompt_type, "math", chat_format=False):
                        fout.write(json.dumps(entry) + "\n")
                with open(output_folder / "validation-sft-chat.jsonl", "wt", encoding="utf-8") as fout:
                    for entry in prepare_for_sft(data, args.prompt_type, "math", chat_format=True):
                        fout.write(json.dumps(entry) + "\n")
            elif args.split_name == "train":
                data = instances[args.validation_size :]
            else:
                data = instances

            output_file = os.path.join(output_folder, f"{args.split_name}.jsonl")
            with open(output_file, "wt", encoding="utf-8") as writer_f:
                for instance in data:
                    writer_f.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    process_data()
