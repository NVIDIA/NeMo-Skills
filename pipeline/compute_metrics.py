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
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from nemo_skills.evaluation.metrics import compute_metrics
from nemo_skills.evaluation.settings import EVALUATOR_MAP
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


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
