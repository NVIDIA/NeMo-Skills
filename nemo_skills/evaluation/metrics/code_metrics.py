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
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "passing_base_tests": prediction['is_correct'],
            "passing_plus_tests": prediction['is_correct-plus'],
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class LiveCodeBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "accuracy": prediction['graded_list'][0],
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
