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
from nemo_skills.evaluation.metrics.utils import is_correct_judgement


class AnswerJudgementMetrics(BaseMetrics):
    def _get_correctness_dict(self, prediction: dict) -> dict[bool]:
        gt_judgement = is_correct_judgement(prediction['expected_judgement'])
        pred_judgement = is_correct_judgement(prediction['judgement'])

        return {'correct_judgements': gt_judgement == pred_judgement}

    def _update_fp_fn(self, metrics_dict, pred_judgement, gt_judgement, divide_by=1):
        is_fp = pred_judgement is True and gt_judgement is False
        is_fn = pred_judgement is False and gt_judgement is True
        metrics_dict['false_positives'] += float(is_fp) / divide_by
        metrics_dict['false_negatives'] += float(is_fn) / divide_by

    def _update_correctness_metrics_for_majority(
        self,
        agg_mode_dict: dict,
        k: int,
        check_correctness_method: str,
        is_correct: bool,
        predictions: list[dict],
        predicted_answers: list[str],
        correctness_dicts: list[dict],
        majority_answer: str,
    ):
        assert check_correctness_method == 'correct_judgements'
        # expected answer is always the same for all predictions, so just take the first one
        gt_judgement = is_correct_judgement(predictions[0]['expected_judgement'])
        self._update_fp_fn(agg_mode_dict[f"majority@{k}"], majority_answer, gt_judgement)

    def _update_correctness_metrics_for_pass(
        self,
        agg_mode_dict: dict,
        k: int,
        check_correctness_method: str,
        is_correct: bool,
        predictions: list[dict],
        correctness_dicts: list[dict],
    ):
        assert check_correctness_method == 'correct_judgements'
        # expected answer is always the same for all predictions, so just take the first one
        gt_judgement = is_correct_judgement(predictions[0]['expected_judgement'])
        pred_judgement = is_correct_judgement(predictions[0]['judgement'])
        # if pass is not correct, means all predictions are the same and wrong
        if not is_correct:
            self._update_fp_fn(agg_mode_dict[f"pass@{k}"], pred_judgement, gt_judgement)

        for pred in predictions[:k]:
            gt_judgement = is_correct_judgement(pred['expected_judgement'])
            pred_judgement = is_correct_judgement(pred['judgement'])
            self._update_fp_fn(agg_mode_dict[f"pass@1[{k}]"], pred_judgement, gt_judgement, divide_by=k)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        predicted_answers = [is_correct_judgement(pred['judgement']) for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        # renaming no_answer to invalid_judgements
        for agg_metric_dict in self.agg_mode_dict.values():
            agg_metric_dict["invalid_judgements"] = agg_metric_dict.pop("no_answer")
        return super().get_metrics()

    def aggregations_to_print(self):
        """We will log all pass/pass@1[k] up to k, but only report the kth one."""
        return [f'pass@1[{self.max_k}]', f'majority@{self.max_k}', f'pass@{self.max_k}']
