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
import glob
import json
import os
import re
from collections import Counter


def validate_code_execution(text, code_begin="```python", code_end="```"):
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        if lines[i] == code_begin:
            code_end_idx = -1
            for j in range(i + 1, len(lines)):
                if lines[j] == code_end:
                    code_end_idx = j
                    break

            if code_end_idx == -1:
                return False

            if code_end_idx + 1 >= len(lines) or lines[code_end_idx + 1] != "```output":
                return False

            output_end_idx = -1
            for j in range(code_end_idx + 2, len(lines)):
                if lines[j] == "```":
                    output_end_idx = j
                    break

            if output_end_idx == -1:
                return False

            i = output_end_idx + 1
        else:
            i += 1

    return True


def cut_final_answer_part(output):
    final_answer_idx = output.find("**Final Answer**")
    if final_answer_idx == -1:
        return None

    boxed_idx = output.find("\\boxed{", final_answer_idx)
    if boxed_idx == -1:
        return None

    balance = 1
    end_idx = boxed_idx + 7
    while balance != 0 and end_idx < len(output):
        if output[end_idx] == "{":
            balance += 1
        elif output[end_idx] == "}":
            balance -= 1
        end_idx += 1

    return output[:end_idx]


def replace_code_tags(text):
    pattern = r"```python\n(.*?)\n```\n"
    replacement = r"<tool_call>\n\1\n</tool_call>\n"
    processed_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return processed_text


def filter_code_solution(sample, args):
    required_keys = ["predicted_answer", "generation", "problem"]
    for key in required_keys:
        if key not in sample:
            return "Key not found: " + key

    # Make some initial filtering to speed up the next llm judgement stage
    if args.code_begin not in sample["generation"]:
        return "No code blocks found"
    if not validate_code_execution(sample["generation"]):
        return "Incomplete code execution found"
    if "judgement" in sample and "judgement: no" in sample["judgement"].lower():
        return "Incorrect final answer"
    if sample["generation"].find("\\boxed{") != -1 and sample["generation"].find("\\boxed{") < sample[
        "generation"
    ].find(args.code_begin):
        return "Boxed before code"
    if sample["generation"].find(sample["predicted_answer"]) != -1 and sample["generation"].find(
        sample["predicted_answer"]
    ) < sample["generation"].find(args.code_begin):
        return "Predicted answer before code"

    generation = cut_final_answer_part(sample["generation"])
    if generation is None:
        return "Final answer not found"

    generation = replace_code_tags(generation)
    sample["generation"] = generation

    return "Accepted"


def preprocess_code_judge(args):
    cnt = Counter()
    with open(args.output_file, "w") as fout:
        for input_file in glob.glob(args.input_files):
            with open(input_file, "r") as fin:
                for idx, line in enumerate(fin):
                    sample = json.loads(line)
                    filt_reason = filter_code_solution(sample, args)
                    cnt[filt_reason] += 1
                    cnt["Total"] += 1
                    if filt_reason is "Accepted":
                        sample["original_index"] = idx
                        fout.write(json.dumps(sample) + "\n")

    print("Filtered samples:")
    for key, value in cnt.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess code judge data")
    parser.add_argument(
        "--input_files", type=str, required=True, help="Input file, could be a pattern like output*.jsonl"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    preprocess_code_judge(args)
