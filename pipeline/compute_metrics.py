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
import logging
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from nemo_skills.utils import setup_logging, unroll_files

LOG = logging.getLogger(__file__)


def fill_up_missing(evaluations, log_probs, pred_answers, allow_incomplete):
    if allow_incomplete:
        evaluations.append(False)
        log_probs.append(-1e9)
        pred_answers.append(None)
    else:
        raise RuntimeError("Need to run evaluate_results.py before computing metrics!")


def compute_metrics(prediction_jsonl_files, allow_incomplete=False, max_samples=-1, aggregation_mode='first'):
    correct_answer = []

    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(prediction_jsonl_files)]

    total_correct = 0
    total_no_answer = 0
    total = 0
    for idx, lines in enumerate(zip_longest(*file_handles)):
        if idx == max_samples:
            break
        evaluations = []
        log_probs = []
        pred_answers = []
        for lidx, line in enumerate(lines):
            if not line:  # could have missing predictions
                fill_up_missing(evaluations, log_probs, pred_answers, allow_incomplete)
                continue
            line_dict = json.loads(line)
            if not line_dict:
                fill_up_missing(evaluations, log_probs, pred_answers, allow_incomplete)
                continue
            if "is_correct" not in line_dict:
                fill_up_missing(evaluations, log_probs, pred_answers, allow_incomplete)
                continue
            pred_answers.append(line_dict["predicted_answer"])
            evaluations.append(line_dict["is_correct"])
            log_probs.append(line_dict.get("log_prob"))
            if line_dict.get("log_prob") is None and aggregation_mode == "most_probable":
                raise RuntimeError(
                    "Cannot compute most probable generation because some of the probabilities are missing"
                )

        total += 1
        if aggregation_mode == "best":
            total_correct += any(evaluations)
            if all([ans is None for ans in pred_answers]):
                total_no_answer += 1
        elif aggregation_mode == "majority":
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers_and_results = [
                (ans, is_correct) for ans, is_correct in zip(pred_answers, evaluations) if ans is not None
            ]
            if len(valid_answers_and_results) == 0:
                total_no_answer += 1
            else:
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                total_correct += majority_result[1]
        elif aggregation_mode == "first":
            total_correct += evaluations[0]
            total_no_answer += pred_answers[0] is None
        elif aggregation_mode == "most_probable":
            most_probable_result = sorted(zip(log_probs, evaluations, pred_answers))[-1]
            total_correct += most_probable_result[1]
            total_no_answer += most_probable_result[2] is None
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    for file_handle in file_handles:
        file_handle.close()

    correct_answer = total_correct / total * 100.0
    no_answer = total_no_answer / total * 100.0
    wrong_answer = (total - total_correct - total_no_answer) / total * 100.0
    return correct_answer, wrong_answer, no_answer, total


if __name__ == '__main__':
    setup_logging(disable_hydra_logs=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_jsonl_files",
        required=True,
        nargs="+",
        help="Can also specify multiple glob patterns, like output-rs*.jsonl",
    )
    parser.add_argument(
        "--save_metrics_file",
        required=False,
        help="Will save metrics here if provided.",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Will allow incomplete evals (e.g. if some of the predictions are missing)",
    )
    parser.add_argument(
        "--max_samples",
        default=-1,
        type=int,
        help="Will cut eval samples at that point (to compare with incomplete evals)",
    )
    parser.add_argument(
        "--aggregation_mode",
        choices=["best", "majority", "first", "most_probable"],
        default="first",
    )
    args = parser.parse_args()

    correct_answer, wrong_answer, no_answer, total = compute_metrics(
        args.prediction_jsonl_files, args.allow_incomplete, args.max_samples, args.aggregation_mode
    )

    LOG.info(f"Evaluation results for %s", args.prediction_jsonl_files)
    LOG.info(f"Total eval entries: %d", total)
    LOG.info(f"Correct answer: %.2f%%", correct_answer)
    LOG.info(f"Wrong answer: %.2f%%", wrong_answer)
    LOG.info(f"No answer: %.2f%%", no_answer)
    if args.save_metrics_file:
        with open(args.save_metrics_file, "wt", encoding="utf-8") as fout:
            json.dump(
                {
                    "num_entries": total,
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "no_answer": no_answer,
                },
                fout,
            )
