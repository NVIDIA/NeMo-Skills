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

        if len(predictions) > 1:
            # Multiple predictions, select the pass@k
            self.agg_mode = f"pass@{len(predictions)}"

            self.correct_proof += any([elem['proof_status'] == "completed" for elem in predictions])
            if all([elem['proof_status'] == "timeout" for elem in predictions]):
                self.timeout_error += 1
        else:
            self.agg_mode = "greedy"

            self.correct_proof += predictions[0]['proof_status'] == "completed"
            self.timeout_error += predictions[0]['proof_status'] == "timeout"

    def get_metrics(self):
        metrics = {"num_entries": self.total}
        metrics["lean4_correct"] = self.correct_proof / self.total * 100.0
        metrics["timeout_error"] = self.timeout_error / self.total * 100.0
        return {self.agg_mode: metrics}

    def reset(self):
        self.correct_proof = 0
        self.timeout_error = 0
        self.total = 0
        # Aggregation mode is automatically set
        self.agg_mode = "greedy"
