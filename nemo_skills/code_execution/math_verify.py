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

import abc
import glob
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

import requests
import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from latex2sympy2_extended.math_normalization import extract_boxed_content
from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import parse, verify, StringExtractionConfig, LatexExtractionConfig, ExprExtractionConfig
from nemo_skills.utils import python_doc_to_cmd_help, unroll_files

LOG = logging.getLogger(__file__)


class DummyFuture:
    def __init__(self, return_value):
        self.return_value = return_value

    def result(self):
        return self.return_value

def unroll_files(input_files):
    for manifest_pattern in input_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def cleanup_tmp_files(input_files):
    # removing any potentially present tmp files
    for manifest in unroll_files(input_files):
        try:
            os.remove(manifest + "-tmp")
        except OSError:
            pass


def dump_data(input_files, data, map_to_future):
    LOG.info("Waiting for current results and dumping to tmp files")
    tmp_file_handles = [
        open(manifest + f"-tmp", "at", encoding="utf-8", buffering=1) for manifest in unroll_files(input_files)
    ]

    for line_data in data:
        for file_data, file_handle in zip(line_data, tmp_file_handles):
            if file_data is None:
                continue
            line_dict = json.loads(file_data)
            if not line_dict:
                file_handle.write("\n")
                continue
            line_dict["is_correct"] = map_to_future[
                (line_dict["predicted_answer"], line_dict["expected_answer"])
            ].result()
            file_handle.write(json.dumps(line_dict) + "\n")

    for file_handle in tmp_file_handles:
        file_handle.close()


def write_tmp_files_back(input_files):
    """Will gracefully handle early exits on errors by properly merging files"""
    LOG.info("Writing temporary files back into original files")
    for manifest in unroll_files(input_files):
        # copying the rest of the results unchanged if any to tmp file
        with open(manifest + "-tmp", "rt") as fin:
            processed_lines = sum(1 for _ in fin)
        with open(manifest, "rt", encoding="utf-8") as fin, open(manifest + "-tmp", "at", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin):
                if line_idx >= processed_lines:
                    fout.write(line)
        # then replacing original file with tmp file
        os.replace(manifest + "-tmp", manifest)


def _additional_normalization(expr):
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".")
    return expr


def verify_answer(gt_answer, predicted_answer, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False
    
    # if we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except:
            predicted_answer = None
        # no need to simpy call in this case
        return predicted_answer == gt_answer

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    if is_mcq:
        parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
        parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
        return verify(parsed_gt, parsed_pred)
    
    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)
    
    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = (re.fullmatch(literal_pattern, normalized_gt) and
                  re.fullmatch(literal_pattern, normalized_pred))

    if is_literal:
        return normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    # Fallback to symbolic comparison
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer
    
    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred, **kwargs)


def batch_evaluate_results(
    input_files: list[str],
    num_parallel_requests=100,
    in_memory_lines=1500,
    numeric_precision=15,
    timeout=10.0,
    take_modulo=None,
    ignore_cache: bool = False,
    use_predicted_answer_key: bool = False,
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
):
    """Will write if the results are correct back into the original files."""

    file_handles = [open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(input_files)]
    cleanup_tmp_files(input_files)

    data = []
    for line_idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
        if line_idx % in_memory_lines == 0:
            if line_idx > 0:  # dumping into tmp files
                dump_data(input_files, data, map_to_future)
            # new in-memory buffer
            data = []
            map_to_future = {}

        data.append([])
        for file_line in lines:
            data[-1].append(file_line)
            if file_line is None:  # if different files have different number of lines
                continue
            line_dict = json.loads(file_line)
            if not line_dict:  # can be empty for incomplete generations
                continue
            gt_answer = line_dict["expected_answer"]

            if not use_predicted_answer_key:
                line_dict["predicted_answer"] = extract_answer(
                    line_dict["generation"],
                    extract_from_boxed=extract_from_boxed,
                    extract_regex=extract_regex,
                )
            else:
                if "predicted_answer" not in line_dict:
                    raise ValueError(
                        "predicted_answer key not found in the line_dict. "
                        "Set use_predicted_answer_key=False to re-extract"
                    )

            data[-1][-1] = json.dumps(line_dict)
            predicted_answer = line_dict["predicted_answer"]

            if (predicted_answer, gt_answer) in map_to_future:
                continue

            if ignore_cache or line_dict.get("is_correct") is None:
                map_to_future[(predicted_answer, gt_answer)] = verify_answer(
                    gt_answer,
                    predicted_answer,
                    take_modulo=take_modulo,
                    numeric_precision=numeric_precision,
                    timeout_seconds=timeout,
                )
            else:
                map_to_future[(predicted_answer, gt_answer)] = DummyFuture(line_dict["is_correct"])

        for file_handle in file_handles:
            file_handle.close()

        if len(data) > 0:
            dump_data(input_files, data, map_to_future)

    write_tmp_files_back(input_files)
