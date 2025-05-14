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

from nemo_skills.evaluation.metrics.base import BaseMetrics
from collections import defaultdict


class Lean4Metrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def get_prediction_results(self, prediction):
        return {
            "correct_proof": prediction['proof_status'] == "completed",
            "timeout_error": prediction['proof_status'] == "timeout",
        }

    def _get_best_prediction(self, stats_dict, elems):
        stats_dict["correct_proof"] += any(elem['correct_proof'] for elem in elems)
        stats_dict["timeout_error"] += all(elem['timeout_error'] for elem in elems)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        self.get_pass_at_k(
            self.agg_mode_dict,
            ['correct_proof', 'timeout_error'],
            predictions,
            pass_at_k_fn=self._get_best_prediction,
        )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, metric_values in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {
                "num_entries": self.total,
                "lean4_correct": (metric_values["correct_proof"] / self.total) * 100.0,
                "timeout_error": (metric_values["timeout_error"] / self.total) * 100.0,
            }
        return metrics_dict

    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))

    def max_aggregations_to_print(self):
        # Return 1 to print only the largest k (or "greedy" and the largest pass@k)
        return 1
