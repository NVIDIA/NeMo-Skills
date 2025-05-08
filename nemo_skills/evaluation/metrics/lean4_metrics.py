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

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        if len(predictions) == 1: # greedy
            self.agg_mode_dict["greedy"]["correct_proof"] += int(predictions[0]['proof_status'] == "completed")
            self.agg_mode_dict["greedy"]["timeout_error"] += int(predictions[0]['proof_status'] == "timeout")


        elif len(predictions) > 1: # pass@k
            # Multiple predictions, select the pass@k
            # getting metrics for all k up to len(predictions). Starting from last to make sure it's printed
            for k in range(len(predictions), 0, -1):
                self.agg_mode_dict[f"pass@{k}"]["correct_proof"] += int(any(elem['proof_status'] == "completed"
                                                              for elem in predictions[:k]))
                self.agg_mode_dict[f"pass@{k}"]["timeout_error"] += int(all(elem['proof_status'] == "timeout"
                                                            for elem in predictions[:k]))


    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, metric_values in self.agg_mode_dict.items():
            metrics = {"num_entries": self.total}
            metrics["lean4_correct"] = (metric_values["correct_proof"] / self.total) * 100.0
            metrics["timeout_error"] = (metric_values["timeout_error"] / self.total) * 100.0
            metrics_dict[agg_mode] = metrics
        return metrics_dict


    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))

    def max_aggregations_to_print(self):
    # Return 1 to print only the largest k (or "greedy" and the largest pass@k)
        return 1
