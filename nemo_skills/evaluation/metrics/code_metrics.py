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


class CodeMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "passing_base_tests": prediction['is_correct'],
            "passing_plus_tests": prediction['is_correct-plus'],
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

    def _get_incorrect_sample(self, predictions, sequence_length):
        for prediction in predictions:
            if 'num_generated_tokens' in prediction and int(prediction['num_generated_tokens']) <= sequence_length: continue
            prediction['passing_base_tests'] = False
            prediction['passing_plus_tests'] = False

class LiveCodeBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "accuracy": prediction['graded_list'][0],
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class SciCodeMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        subtask_status_list = prediction['eval_status']
        correct_subtasks = sum(subtask['process_status'] == 'completed' for subtask in subtask_status_list)
        return {
            'problem_accuracy': correct_subtasks == len(subtask_status_list),
            'subtask_accuracy': correct_subtasks,
        }

    def update(self, predictions):
        super().update(predictions)
        self.subtasks_total += len(predictions[0]['eval_status'])
        self._compute_pass_at_k(predictions)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for agg_mode in self.eval_dict.keys():
            metrics_dict[agg_mode]["num_problems"] = metrics_dict[agg_mode].pop("num_entries")
            metrics_dict[agg_mode]["num_subtasks"] = self.subtasks_total
            # correcting subtask normalization
            metrics_dict[agg_mode]["subtask_accuracy"] *= self.total / self.subtasks_total

        return metrics_dict

    def reset(self):
        super().reset()
        self.subtasks_total = 0
