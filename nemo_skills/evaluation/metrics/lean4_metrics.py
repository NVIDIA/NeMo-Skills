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


class Lean4Metrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def _get_correctness_dict(self, prediction):
        return {"lean4_correct": prediction['proof_status'] == "completed"}

    def _update_correctness_metrics_for_pass(
        self,
        agg_mode_dict: dict,
        k: int,
        check_correctness_method: str,
        pass_score: bool | float | int,
        predictions: list[dict],
        correctness_dicts: list[dict],
    ):
        assert check_correctness_method == 'lean4_correct'
        timeout_errors = [pred['proof_status'] == "timeout" for pred in predictions[:k]]
        agg_mode_dict[f'pass@{k}']['timeout_error'] += all(timeout_errors)
        agg_mode_dict[f'pass@1[{k}]']['timeout_error'] += sum(timeout_errors) / k

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
