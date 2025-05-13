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

from collections import Counter, defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement


class AnswerJudgementMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def update_perf_dict(self, perf_dict, is_correct, is_fp, is_fn):
        perf_dict["total_correct"] += int(is_correct)
        perf_dict["fp_count"] += int(is_fp)
        perf_dict["fn_count"] += int(is_fn)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        if len(predictions) > 1:
            # Majority@k
            # Reinitialize local vars
            is_correct, is_fp, is_fn = False, False, False

            answers = [is_correct_judgement(elem['judgement']) for elem in predictions]
            majority_judgement = Counter(answers).most_common(1)[0][0]
            is_correct = majority_judgement == is_correct_judgement(predictions[0]['expected_judgement'])

            if not is_correct:
                if majority_judgement:
                    is_fp = True
                else:
                    is_fn = True

            self.update_perf_dict(self.agg_mode_dict[f"majority@{len(predictions)}"], is_correct, is_fp, is_fn)

            # Pass@k
            is_correct, is_fp, is_fn = False, False, False
            is_correct = any(
                [
                    is_correct_judgement(elem['judgement']) == is_correct_judgement(elem['expected_judgement'])
                    for elem in predictions
                ]
            )

            if not is_correct:
                if is_correct_judgement(predictions[0]['judgement']):
                    is_fp = True
                else:
                    is_fn = True

            self.update_perf_dict(self.agg_mode_dict[f"pass@{len(predictions)}"], is_correct, is_fp, is_fn)
        else:
            is_fp, is_fn = False, False
            is_correct = is_correct_judgement(predictions[0]['judgement']) == is_correct_judgement(
                predictions[0]['expected_judgement']
            )
            if not is_correct:
                if is_correct_judgement(predictions[0]['judgement']):
                    is_fp = True
                else:
                    is_fn = True

            self.update_perf_dict(self.agg_mode_dict[f"greedy"], is_correct, is_fp, is_fn)

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}

            metrics_dict[agg_mode]["correct_judgements"] = (agg_metric_dict["total_correct"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_positives"] = (agg_metric_dict["fp_count"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_negatives"] = (agg_metric_dict["fn_count"] / self.total) * 100.0

        return metrics_dict

    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))
