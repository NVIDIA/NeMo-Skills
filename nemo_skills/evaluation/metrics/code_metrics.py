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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CodeMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def get_prediction_results(self, prediction):
        return {
            "total_correct": prediction['is_correct'],
            "total_correct_plus": prediction['is_correct-plus'],
        }

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        self.get_pass_at_k(self.agg_mode_dict, predictions=predictions)

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {
                "num_entries": self.total,
                "passing_base_tests": agg_metric_dict["total_correct"] / self.total * 100.0,
                "passing_plus_tests": agg_metric_dict["total_correct_plus"] / self.total * 100.0,
            }

        return metrics_dict

    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(float))

    def max_aggregations_to_print(self):
        """We will log all pass/pass@1[k] up to k, but only report the kth one."""
        # pass + pass@1[k]
        return 1 + 1
