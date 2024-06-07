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

import re
from collections import defaultdict
from typing import List

from nemo_skills.code_execution import CODE_OUTPUT_SEPARATORS, CODE_SEPARATORS


def process_bad_solutions(
    samples: List, solution_filters: List[str], text_filter_type: str, should_trim: bool
) -> List:
    """
    Apply filters and trimming to the list of solutions `samples`.
    `samples` is a list of solutions for a single problem.
    """
    filtered_predictions = []
    code_solns = []
    text_solns = []
    for sample in samples:
        if should_remove(sample['generation'], solution_filters):
            continue
        if should_trim:
            sample['generation'] = trim_output(sample['generation'])
        if CODE_SEPARATORS[0] in sample['generation']:
            code_solns.append(sample)
        else:
            text_solns.append(sample)
    if text_filter_type is None:
        filtered_predictions.extend(code_solns)
        filtered_predictions.extend(text_solns)
    elif text_filter_type == 'all':
        filtered_predictions.extend(code_solns)
    elif text_filter_type == 'majority_code':
        filtered_predictions.extend(code_solns)
        if len(code_solns) <= len(samples) // 2:
            filtered_predictions.extend(text_solns)
    elif text_filter_type == 'majority_text':
        if len(code_solns) > len(samples) // 2:
            filtered_predictions.extend(code_solns)
        else:
            filtered_predictions.extend(text_solns)
    elif text_filter_type == 'any_code':
        if code_solns:
            filtered_predictions.extend(code_solns)
        else:
            filtered_predictions.extend(text_solns)
    else:
        raise NotImplementedError(f"Filtering method {text_filter_type} not implemented")

    return filtered_predictions


def downsample_data(input_instances: List, sampling_method: str, num_samples):
    """
    Downsample data to a given size using either 'random' or 'fair' method.

    - 'random': Takes the first 'num_samples' samples assuming the data is already shuffled.
    - 'fair': Balances the number of solutions per problem by selecting them one-by-one.
    """
    output_instances = []
    if sampling_method == "random":
        output_instances = input_instances[:num_samples]
    elif sampling_method == "fair":
        question_to_solns = defaultdict(list)
        for instance in input_instances:
            question = instance["input"]
            question_to_solns[question].append(instance)

        soln_counter = 0
        questions = list(question_to_solns.keys())
        while True:
            for quesn in questions:
                if len(output_instances) == num_samples:
                    break
                if len(question_to_solns[quesn]) > soln_counter:
                    output_instances.append(question_to_solns[quesn][soln_counter])
            soln_counter += 1
            if len(output_instances) == num_samples:
                break
    else:
        raise NotImplementedError(f"Sampling method {sampling_method} not implemented")

    return output_instances


PATTERN_ANS = re.compile(r"\\boxed\{([^}]*)\}")
PATTERN_CODE = re.compile(CODE_SEPARATORS[0])


def remove_multi_boxed(generation: str):
    """
    Returns `True` if the solution contains more than one `\\boxed` entry.
    """
    if len(PATTERN_ANS.findall(generation)) > 1:
        return True

    return False


def remove_broken_code(generation: str):
    """
    Returns `True` if the solution has inconsistent code and code output separators order.
    """
    code_start_indices = [match.start() for match in re.finditer(CODE_SEPARATORS[0], generation)]
    code_end_indices = [match.start() for match in re.finditer(CODE_SEPARATORS[1], generation)]
    code_out_start_indices = [match.start() for match in re.finditer(CODE_OUTPUT_SEPARATORS[0], generation)]
    code_out_end_indices = [match.start() for match in re.finditer(CODE_OUTPUT_SEPARATORS[1], generation)]

    num_code_occs = set(
        [len(code_start_indices), len(code_end_indices), len(code_out_start_indices), len(code_out_end_indices)]
    )
    if len(num_code_occs) != 1:
        return True

    if not len(code_end_indices):
        return False

    for code_start_idx, code_end_idx, code_out_start_idx, code_out_end_idx in zip(
        code_start_indices, code_end_indices, code_out_start_indices, code_out_end_indices
    ):
        if not (code_start_idx < code_end_idx < code_out_start_idx < code_out_end_idx):
            return True

    return False


def remove_useless_code(generation: str):
    """
    Returns `True` if the solution has the first `\\boxed` entry before the first code block.
    """
    ans_match = PATTERN_ANS.search(generation)
    code_match = PATTERN_CODE.search(generation)
    if not ans_match or not code_match:
        return False

    return ans_match.start() < code_match.start()


filters_map = {
    "multi_boxed": remove_multi_boxed,
    "broken_code": remove_broken_code,
    "useless_code": remove_useless_code,
}


def should_remove(generation: str, filters: List):
    """
    Applies filters sequentially to a generated solution.
    Returns `True` if any filter returns `True`.
    """
    for filter_name in filters:
        filter_fn = filters_map[filter_name]
        if filter_fn(generation):
            return True

    return False


def trim_output(output_string: str) -> str:
    """
    Removes everything after the last occurrence of `\\boxed` in the output.
    """
    output_lines = output_string.split("\n")

    stop_idx = 0
    for idx, soln_line in enumerate(output_lines):
        if PATTERN_ANS.findall(soln_line):
            stop_idx = idx
            break

    if stop_idx < len(output_lines) - 1 and (
        "\\end{align" in output_lines[stop_idx + 1]
        or "\]" in output_lines[stop_idx + 1]
        or "$$" in output_lines[stop_idx + 1]
    ):
        stop_idx = stop_idx + 1

    trimmed_output = "\n".join(output_lines[: stop_idx + 1])

    return trimmed_output
