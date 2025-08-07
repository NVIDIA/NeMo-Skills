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

import logging
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class HLEMetrics(BaseMetrics):
    # TODO: how can we ensure that user-defined aggregations have all the same metrics as in base?
    @classmethod
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        correctness_dict = {}
        if 'judgement' in prediction:
            correctness_dict["judge_correct"] = is_correct_judgement(prediction['judgement'])

        return correctness_dict

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        prediction = prediction.copy()
        if 'judgement' in prediction:
            prediction['judgement'] = 'Judgement: No'
        return prediction

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        predicted_answers = [pred['generation'] for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        if 'reward_model_score' in predictions[0]:
            self._compute_reward_at_k(predictions=predictions)


    def evaluations_to_print(self):
        """We will log all majority/rm/pass/pass@1[avg-of-k] up to k, but only report the kth one."""
        return [
            f'pass@1[avg-of-{self.max_k}]',
            f'majority@{self.max_k}',
            f'rm_best@{self.max_k}',
            f'rm_majority@{self.max_k}',
            f'pass@{self.max_k}',
        ]

    def metrics_to_print(self):
        return {
            'num_entries': as_int,
            'avg_tokens': as_int,
            'gen_seconds': as_int,
            'judge_correct': as_percentage,
            'symbolic_correct': as_percentage,
            'no_answer': as_percentage,
        }
