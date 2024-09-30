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
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from fix_ref_solns import _fix_solution, _post_fix, _post_fix_multi_answer

from nemo_skills.code_execution.math_grader import normalize_answer_string

DOWNLOAD_LINK = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"


def find_boxed_entries(answer_str):
    stack = []
    results = []
    i = 0

    while i < len(answer_str):
        if answer_str[i : i + 7] == '\\boxed{':
            stack.append(i + 7)
            i += 7
        elif answer_str[i] == '{':
            if stack:
                stack.append(i + 1)
            i += 1
        elif answer_str[i] == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    results.append(answer_str[start:i])
            i += 1
        else:
            i += 1

    if len(results) == 0:
        raise ValueError("Not enough boxed entries")
    else:
        results = [normalize_answer_string(result) for result in results]

    if len(results) == 1:
        # Single boxed entry, trivial case
        return results

    else:
        # Multiple boxed entries. There are two cases possible
        # (a) The reference solution has the same question answered in multiple ways
        # (b) The answer is split across multiple boxed entries and we need to merge
        result_equal = True
        for idx in range(len(results) - 1):
            if not (results[idx] == results[idx + 1]):
                result_equal = False
                break

        if result_equal:
            # Same problem solved in multiple ways
            return [results[0]]
        else:
            return results


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


def save_data(split, random_seed, validation_size):
    output_dir = Path(__file__).absolute().parent
    output_dir.mkdir(exist_ok=True)
    actual_split = "test" if split == "test" else "train"

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
                if eval_set != actual_split:
                    continue

                content = json.loads(reader_f.extractfile(tar_member).read())
                content["id"] = f"{eval_set}/{problem_type}/{fileid}.json"
                # Load the solution with our identified fixes
                content["reference_solution"] = _fix_solution(content["id"], content["solution"])
                del content["solution"]

                entries = find_boxed_entries(content["reference_solution"])
                if len(entries) == 1:
                    parsed_answer = entries[0]
                if len(entries) > 1:
                    parsed_answer = _post_fix_multi_answer(content["id"], entries)

                if not (
                    ("Find the equation" in content["problem"])
                    or ("Enter the equation" in content["problem"])
                    or ("What is the equation") in content["problem"]
                    or ("described by the equation") in content["problem"]
                    or ("Find an equation") in content["problem"]
                ) and ("=" in parsed_answer):
                    if parsed_answer.count("=") == 1:
                        # For greater count, it means we're just predicting values of multiple variables
                        parsed_answer = parsed_answer.split("=")[1]
                content["expected_answer"] = parsed_answer

                # Sanity check that content type matches the parent dir
                content_type = content["type"].lower()
                content_type = content_type.replace(" ", "_")
                content_type = content_type.replace("&", "and")
                assert problem_type == content_type

                content["expected_answer"] = _post_fix(content["id"], content["expected_answer"])

                split_instances_dict[eval_set].append(content)

        assert len(split_instances_dict) == 1
        for split, instances in split_instances_dict.items():
            # always shuffling to make it easier to get validation/train out of train_full
            if split != "test":
                random.seed(random_seed)
                random.shuffle(instances)
            if split == "validation":
                data = instances[:validation_size]
            elif split == "train":
                data = instances[validation_size:]
            else:
                data = instances

            output_file = os.path.join(output_dir, f"{split}.jsonl")
            with open(output_file, "wt", encoding="utf-8") as writer_f:
                for instance in data:
                    writer_f.write(json.dumps(instance) + "\n")


def process_data():
    """Download tar and condense data into single jsonl file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "test", "validation", "train", "train_full"),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=1000)
    args = parser.parse_args()

    if args.split == "all":
        for split in ["test", "validation", "train", "train_full"]:
            save_data(split, args.random_seed, args.validation_size)
    else:
        save_data(args.split, args.random_seed, args.validation_size)


if __name__ == "__main__":
    process_data()
