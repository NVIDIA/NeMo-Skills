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


class IFMetrics(BaseMetrics):
    # loosely adapted from
    # https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_main.py

    required_keys = ['follow_instruction_list', 'instruction_id_list']

    def __init__(self):
        self.reset()

    def get_prediction_results(self, prediction):
        """Extract metrics from a prediction."""
        return {
            'prompt': prediction['follow_all_instructions'],
            'instruction': sum(prediction['follow_instruction_list']),
            'follow_instruction_list': prediction['follow_instruction_list'],
        }

    def _get_best_prediction(self, stats_dict, elems):
        """Will update using the pass@k strategy (just pass a single-element list to get greedy)."""
        # computing "pass@k" score
        follow_instruction_list = elems[0]['follow_instruction_list']
        for elem in elems:
            follow_instruction_list = [
                follow_instruction_list[i] or elem['follow_instruction_list'][i]
                for i in range(len(follow_instruction_list))
            ]

        if all(follow_instruction_list):
            stats_dict['prompt'] += 1

        stats_dict['instruction'] += sum(follow_instruction_list)

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1
        self.instruction_total += len(predictions[0]['instruction_id_list'])

        strict_predictions = [pred['strict_eval'] for pred in predictions]
        loose_predictions = [pred['loose_eval'] for pred in predictions]

        self.get_pass_at_k(
            self.strict_agg_mode_dict,
            pred_keys=['prompt', 'instruction'],
            predictions=strict_predictions,
            pass_at_k_fn=self._get_best_prediction,
        )
        self.get_pass_at_k(
            self.loose_agg_mode_dict,
            pred_keys=['prompt', 'instruction'],
            predictions=loose_predictions,
            pass_at_k_fn=self._get_best_prediction,
        )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode in self.strict_agg_mode_dict:
            prompt_strict = self.strict_agg_mode_dict[agg_mode]['prompt'] / self.total * 100.0
            inst_strict = self.strict_agg_mode_dict[agg_mode]['instruction'] / self.instruction_total * 100.0
            prompt_loose = self.loose_agg_mode_dict[agg_mode]['prompt'] / self.total * 100.0
            inst_loose = self.loose_agg_mode_dict[agg_mode]['instruction'] / self.instruction_total * 100.0
            metrics_dict[agg_mode] = {
                "num_prompts": self.total,
                "num_instructions": self.instruction_total,
                "average_score": (prompt_strict + inst_strict + prompt_loose + inst_loose) / 4,
                "prompt_strict_accuracy": prompt_strict,
                "instruction_strict_accuracy": inst_strict,
                "prompt_loose_accuracy": prompt_loose,
                "instruction_loose_accuracy": inst_loose,
            }

        return metrics_dict

    def reset(self):
        self.total = 0
        self.instruction_total = 0
        self.strict_agg_mode_dict = defaultdict(lambda: {"prompt": 0.0, "instruction": 0.0})
        self.loose_agg_mode_dict = defaultdict(lambda: {"prompt": 0.0, "instruction": 0.0})

    def max_aggregations_to_print(self):
        """We will log all pass/pass@1[k] up to k, but only report the kth one."""
        # pass + pass@1[k]
        return 1 + 1
