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


class MathEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'predicted_answer': None, 'is_correct': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'predicted_answer' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            if all([elem['predicted_answer'] is None for elem in predictions]):
                self.total_no_answer += 1
        elif aggregation_mode == "majority":
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers_and_results = [
                (elem['predicted_answer'], elem['is_correct'])
                for elem in predictions
                if elem['predicted_answer'] is not None
            ]
            if len(valid_answers_and_results) == 0:
                self.total_no_answer += 1
            else:
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                self.total_correct += majority_result[1]
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_no_answer += predictions[0]['predicted_answer'] is None
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "correct_answer": self.total_correct / self.total * 100.0,
            "wrong_answer": (self.total - self.total_correct - self.total_no_answer) / self.total * 100.0,
            "no_answer": self.total_no_answer / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_no_answer = 0
        self.total = 0


class CodeEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'is_correct': False, 'is_correct-plus': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'is_correct-plus' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            self.total_correct_plus += any([elem['is_correct-plus'] for elem in predictions])
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_correct_plus += predictions[0]['is_correct-plus']
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "passing_base_tests": self.total_correct / self.total * 100.0,
            "passing_plus_tests": self.total_correct_plus / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_correct_plus = 0
        self.total = 0


class IFEval:
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'loose_eval': {'follow_all_instructions': False}, 'strict_eval': {'follow_all_instructions': False}}

    def is_incomplete(self, elem):
        incomplete = 'loose_eval' not in elem or 'strict_eval' not in elem
        if incomplete:
            return False
        return (
            'follow_all_instructions' not in elem['loose_eval'] or 'follow_all_instructions' not in elem['strict_eval']
        )

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct_loose += any([elem['loose_eval']['follow_all_instructions'] for elem in predictions])
            self.total_correct_strict += any([elem['strict_eval']['follow_all_instructions'] for elem in predictions])
        elif aggregation_mode == "first":
            self.total_correct_loose += predictions[0]['loose_eval']['follow_all_instructions']
            self.total_correct_strict += predictions[0]['strict_eval']['follow_all_instructions']
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "strict_accuracy": self.total_correct_strict / self.total * 100.0,
            "loose_accuracy": self.total_correct_loose / self.total * 100.0,
        }

    def reset(self):
        self.total_correct_loose = 0
        self.total_correct_strict = 0
        self.total = 0


def compute_metrics(
    prediction_jsonl_files,
    evaluator,
    allow_incomplete=False,
    max_samples=-1,
    aggregation_mode='first',
):
    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(prediction_jsonl_files)]

    evaluator.reset()
    for idx, lines in enumerate(zip_longest(*file_handles)):
        if idx == max_samples:
            break
        data = []
        for line in lines:
            if not line:  # could have missing predictions
                if not allow_incomplete:
                    raise RuntimeError("Some data is missing!")
                data.append(evaluator.fill_up_missing())
                continue
            line_dict = json.loads(line)
            if not line_dict:
                if not allow_incomplete:
                    raise RuntimeError("Some data is missing!")
                data.append(evaluator.fill_up_missing())
                continue
            if evaluator.is_incomplete(line_dict):
                if not allow_incomplete:
                    raise RuntimeError("Some data is missing!")
                data.append(evaluator.fill_up_missing())
                continue
            data.append(line_dict)

        evaluator.update(data, aggregation_mode)

    for file_handle in file_handles:
        file_handle.close()

    return evaluator.get_metrics()


math_benchmarks = [
    'algebra222',
    'asdiv',
    'functional',
    'gsm-hard',
    'gsm-ic-2step',
    'gsm-ic-mstep',
    'gsm-plus',
    'gsm8k',
    'math',
    'mawps',
    'svamp',
    'tabmwp',
]
code_benchmarks = ['human-eval', 'mbpp']

EVALUATOR_MAP = {
    "ifeval": IFEval,
    "mmlu": MathEval,  # TODO: update this
}

for benchmark in math_benchmarks:
    EVALUATOR_MAP[benchmark] = MathEval

for benchmark in code_benchmarks:
    EVALUATOR_MAP[benchmark] = CodeEval


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
        "--benchmark",
        required=True,
        help="To select which evaluator to use",
    )
    parser.add_argument(
        "--aggregation_mode",
        choices=["best", "majority", "first"],
        default="first",
    )
    args = parser.parse_args()

    evaluator = EVALUATOR_MAP[args.benchmark]()

    metrics = compute_metrics(
        args.prediction_jsonl_files,
        evaluator,
        args.allow_incomplete,
        args.max_samples,
        args.aggregation_mode,
    )

    LOG.info(f"Evaluation results for %s", args.prediction_jsonl_files)
    for metric_key, metric_value in metrics.items():
        if isinstance(metric_value, float):
            metric_value = f"{metric_value:.2f}"
        LOG.info(f"%s: %s", metric_key, metric_value)
    if args.save_metrics_file:
        with open(args.save_metrics_file, "wt", encoding="utf-8") as fout:
            json.dump(metrics, fout, indent=4)
