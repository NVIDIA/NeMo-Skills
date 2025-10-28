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

from nemo_skills.evaluation.metrics.base import BaseMetrics, default_formatting


class NemoEvaluatorMetrics(BaseMetrics):
    """Metrics translator from the format of Nemo-Evaluator.

    WIPP: explain the structure
    """

    def get_metrics(self):
        return {
            "_main_evaluation_mode": {
                "some_metric": 42,
            },
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"is_correct": True}

    def update(self, predictions):
        super().update(predictions)

    def metrics_to_print(self):
        """Control which metrics are displayed in the summary table."""

        return {
            "some_metric": default_formatting,
        }

    def evaluations_to_print(self):
        return ["_main_evaluation_mode"]
