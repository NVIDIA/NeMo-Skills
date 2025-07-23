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
from functools import partial

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement

is_correct_judgement_or_none = partial(is_correct_judgement, return_none=True)


class AnswerJudgementMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        # Store individual TP/FP/FN/TN values as N x K matrix (N datapoints, K samples each)
        self.total_positives = 0
        self.individual_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def reset(self):
        super().reset()
        self.individual_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        gt_judgement = is_correct_judgement_or_none(prediction['expected_judgement'])
        pred_judgement = is_correct_judgement_or_none(prediction['judgement'])

        return {'correct_judgements': gt_judgement == pred_judgement}

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        prediction = prediction.copy()
        if is_correct_judgement_or_none(prediction['expected_judgement']):
            prediction["predicted_judgement"] = "Judgement: No"
        else:
            prediction["predicted_judgement"] = "Judgement: Yes"
        return prediction

    def _store_individual_metrics(self, agg_key, pred_judgement, gt_judgement, sample_idx=0):
        """Store individual TP/FP/FN/TN values in N x K matrix structure."""
        is_fp = pred_judgement is True and gt_judgement is False
        is_fn = pred_judgement is False and gt_judgement is True
        is_tp = pred_judgement is True and gt_judgement is True
        is_tn = pred_judgement is False and gt_judgement is False

        # Store in N x K matrix: [datapoint_idx][sample_idx]
        # This is hacky, but the only way to access the datapoint_idx
        datapoint_idx = self.total - 1
        self.individual_metrics[agg_key][datapoint_idx][sample_idx] = {
            'tp': float(is_tp),
            'fp': float(is_fp),
            'fn': float(is_fn),
            'tn': float(is_tn),
        }

    def _update_fp_fn(self, metrics_dict, pred_judgement, gt_judgement, divide_by=1):
        is_fp = pred_judgement is True and gt_judgement is False
        is_fn = pred_judgement is False and gt_judgement is True

        metrics_dict['false_positives'] += float(is_fp) / divide_by
        metrics_dict['false_negatives'] += float(is_fn) / divide_by

    def _update_score_metrics_for_majority(
        self,
        eval_dict: dict,
        k: int,
        score_method: str,
        score_dicts: list[dict],
        majority_score: bool | float | int,
        majority_answer: str,
        predictions: list[dict],
        predicted_answers: list[str],
    ):
        assert score_method == 'correct_judgements'
        # expected answer is always the same for all predictions, so just take the first one
        gt_judgement = is_correct_judgement_or_none(predictions[0]['expected_judgement'])
        self._update_fp_fn(eval_dict[f"majority@{k}"], majority_answer, gt_judgement)
        self._store_individual_metrics(f"majority@{k}", majority_answer, gt_judgement)

    def _update_score_metrics_for_pass(
        self,
        eval_dict: dict,
        k: int,
        score_method: str,
        score_dicts: list[dict],
        pass_score: bool | float | int,
        predictions: list[dict],
        predicted_answers: list[str] | None,
    ):
        assert score_method == 'correct_judgements'
        # expected answer is always the same for all predictions, so just take the first one
        gt_judgement = is_correct_judgement_or_none(predictions[0]['expected_judgement'])
        pred_judgements = [is_correct_judgement_or_none(pred['judgement']) for pred in predictions[:k]]
        if gt_judgement in pred_judgements:
            pred_judgement = gt_judgement
        else:
            not_none_pred_judgements = [
                pred_judgement for pred_judgement in pred_judgements if pred_judgement is not None
            ]
            pred_judgement = not_none_pred_judgements[0] if not_none_pred_judgements else None

        self._update_fp_fn(eval_dict[f"pass@{k}"], pred_judgement, gt_judgement)
        self._store_individual_metrics(f"pass@{k}", pred_judgement, gt_judgement)

        for sample_idx, pred in enumerate(predictions[:k]):
            gt_judgement = is_correct_judgement_or_none(pred['expected_judgement'])
            pred_judgement = is_correct_judgement_or_none(pred['judgement'])
            self._update_fp_fn(eval_dict[f"pass@1[avg-of-{k}]"], pred_judgement, gt_judgement, divide_by=k)
            self._store_individual_metrics(f"pass@1[avg-of-{k}]", pred_judgement, gt_judgement, sample_idx)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        self.total_positives += float(is_correct_judgement_or_none(predictions[0]['expected_judgement']) is True)
        predicted_answers = [is_correct_judgement_or_none(pred['judgement']) for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def _compute_precision_recall_f1(self, datapoint_metrics):
        """Compute unbiased precision, recall, F1 by averaging over K samples."""
        # Find the maximum number of samples K across all datapoints
        max_k = max(len(sample_metrics) for sample_metrics in datapoint_metrics.values())

        # Compute metrics for each of the K samples, then average across K
        sample_precision_values = []
        sample_recall_values = []
        sample_f1_values = []

        for sample_idx in range(max_k):
            # Aggregate TP, FP, FN across all N datapoints for sample k
            total_tp, total_fp, total_fn = 0, 0, 0

            for sample_metrics in datapoint_metrics.values():
                metrics = sample_metrics[sample_idx]
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']

            # Compute precision for sample k
            if total_tp + total_fp > 0:
                sample_precision = total_tp / (total_tp + total_fp)
            else:
                sample_precision = 1.0
            sample_precision_values.append(sample_precision)

            # Compute recall for sample k
            if self.total_positives > 0:
                sample_recall = total_tp / self.total_positives
            else:
                sample_recall = 1.0
            sample_recall_values.append(sample_recall)

            # Compute F1 for sample k
            if sample_precision + sample_recall > 0:
                sample_f1 = 2 * (sample_precision * sample_recall) / (sample_precision + sample_recall)
            else:
                sample_f1 = 0.0
            sample_f1_values.append(sample_f1)

        # Average across all K samples
        return {
            'precision': 100 * sum(sample_precision_values) / max_k,
            'recall': 100 * sum(sample_recall_values) / max_k,
            'f1': 100 * sum(sample_f1_values) / max_k,
        }

    def get_metrics(self):
        # renaming no_answer to invalid_judgements
        for agg_metric_dict in self.eval_dict.values():
            agg_metric_dict["invalid_judgements"] = agg_metric_dict.pop("no_answer")

        metrics_dict = super().get_metrics()

        # Compute unbiased precision, recall, F1 by averaging over K samples
        for agg_key, datapoint_metrics in self.individual_metrics.items():
            if agg_key in metrics_dict:
                metrics_dict[agg_key].update(self._compute_precision_recall_f1(datapoint_metrics))
        return metrics_dict
