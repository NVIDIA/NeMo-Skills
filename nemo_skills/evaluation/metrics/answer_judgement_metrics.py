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

    def get_prediction_results(self, prediction):
        gt_judgement = is_correct_judgement(prediction['expected_judgement'])
        pred_judgement = is_correct_judgement(prediction['judgement'])
        return {
            'gt_judgement': gt_judgement,
            'pred_judgement': pred_judgement,
            'is_correct': pred_judgement == gt_judgement,
            'is_invalid': pred_judgement is None,
            'is_fp': pred_judgement == True and gt_judgement == False,
            'is_fn': pred_judgement == False and gt_judgement == True,
        }

    def _get_best_prediction(self, agg_mode_dict, prediction_results):
        gt_judgement = prediction_results[0]['gt_judgement']
        valid_answers = [elem['pred_judgement'] for elem in prediction_results if not elem['is_invalid']]
        if len(valid_answers) == 0:
            return

        pred_answer = valid_answers[0]
        for answer in valid_answers:
            if answer == gt_judgement:
                pred_answer = answer
                break

        agg_mode_dict['is_correct'] += float(pred_answer == gt_judgement)
        agg_mode_dict['is_fp'] += float(pred_answer == True and gt_judgement == False)
        agg_mode_dict['is_fn'] += float(pred_answer == False and gt_judgement == True)
        agg_mode_dict['is_invalid'] += float(pred_answer is None)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        prediction_results = [self.get_prediction_results(prediction) for prediction in predictions]

        # Greedy
        if len(predictions) == 1:
            self.get_pass_at_k(
                self.agg_mode_dict,
                pred_keys=['is_correct', 'is_fp', 'is_fn', 'is_invalid'],
                prediction_results=prediction_results,
            )
        else:
            # Majority@k, Pass@k, Pass@1[k]
            self.get_pass_at_k(
                self.agg_mode_dict,
                pred_keys=['is_correct', 'is_fp', 'is_fn', 'is_invalid'],
                prediction_results=prediction_results,
                pass_at_k_fn=self._get_best_prediction,
            )
            self.get_majority_at_k(
                self.agg_mode_dict,
                predicted_answers=[pred['pred_judgement'] for pred in prediction_results],
                pred_keys=['is_correct', 'is_fp', 'is_fn', 'is_invalid'],
                prediction_results=prediction_results,
            )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}

            metrics_dict[agg_mode]["correct_judgements"] = (agg_metric_dict["is_correct"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_positives"] = (agg_metric_dict["is_fp"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_negatives"] = (agg_metric_dict["is_fn"] / self.total) * 100.0
            metrics_dict[agg_mode]["invalid_judgements"] = (agg_metric_dict["is_invalid"] / self.total) * 100.0

        return metrics_dict

    def aggregations_to_print(self):
        """We will log all pass/pass@1[k] up to k, but only report the kth one."""
        # majority + pass + pass@1[k]
        return [f'majority@{self.max_k}', f'pass@{self.max_k}', f'pass@1[{self.max_k}]']
