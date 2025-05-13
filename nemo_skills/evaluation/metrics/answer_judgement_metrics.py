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
from typing import Union
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement


class AnswerJudgementMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def update_perf_dict(self, perf_dict, is_correct, is_fp, is_fn, invalid_count):
        perf_dict["total_correct"] += float(is_correct)
        perf_dict["fp_count"] += float(is_fp)
        perf_dict["fn_count"] += float(is_fn)
        perf_dict["invalid_count"] += float(invalid_count)
    
    def get_judgement_by_type(self, predictions, judgement_type: str, gt_judgement: bool) -> Union[bool, None]:
        answers = [c for elem in predictions if (c:=is_correct_judgement(elem['judgement'])) is not None]
        if len(answers) == 0:
            return None
        if judgement_type == "majority":
            return Counter(answers).most_common(1)[0][0]
        elif judgement_type == "pass":
            for answer in answers:
                if answer == gt_judgement:
                    return answer
            return answers[0]
        else:
            raise ValueError(f"Invalid judgement type: {judgement_type}")
    
    def get_judgement_metrics(self, pred_judgement, gt_judgement):
        is_fp, is_fn = False, False
        is_invalid = pred_judgement is None
        is_correct = pred_judgement == gt_judgement
        if not is_correct:
            if pred_judgement == True:
                is_fp = True
            elif pred_judgement == False:
                is_fn = True
        return is_correct, is_fp, is_fn, is_invalid
        

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        gt_judgement = is_correct_judgement(predictions[0]['expected_judgement'])
        if len(predictions) > 1:
            # Majority@k
            # Reinitialize local vars
            majority_judgement = self.get_judgement_by_type(predictions, "majority", gt_judgement)
            majority_metrics = self.get_judgement_metrics(majority_judgement, gt_judgement)
            self.update_perf_dict(self.agg_mode_dict[f"majority@{len(predictions)}"], *majority_metrics)

            # Pass@k
            pass_judgement = self.get_judgement_by_type(predictions, "pass", gt_judgement)
            pass_metrics = self.get_judgement_metrics(pass_judgement, gt_judgement)
            self.update_perf_dict(self.agg_mode_dict[f"pass@{len(predictions)}"], *pass_metrics)
        
        # greedy for single sample, pass@1[k] for multiple samples
        single_sample_metric_name = "greedy" if len(predictions) == 1 else f"pass@1[{len(predictions)}]"
        per_sample_metrics = [self.get_judgement_metrics(is_correct_judgement(prediction['judgement']), gt_judgement) for prediction in predictions]
        avg_per_sample_metrics = [sum(metrics) / len(metrics) for metrics in zip(*per_sample_metrics)]
        self.update_perf_dict(self.agg_mode_dict[single_sample_metric_name], *avg_per_sample_metrics)


    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}

            metrics_dict[agg_mode]["correct_judgements"] = (agg_metric_dict["total_correct"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_positives"] = (agg_metric_dict["fp_count"] / self.total) * 100.0
            metrics_dict[agg_mode]["false_negatives"] = (agg_metric_dict["fn_count"] / self.total) * 100.0
            metrics_dict[agg_mode]["invalid_judgements"] = (agg_metric_dict["invalid_count"] / self.total) * 100.0

        return metrics_dict

    def reset(self):
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))
