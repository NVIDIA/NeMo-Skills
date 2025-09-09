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


class IOIMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, predictions):
        super().update(predictions)
        if len(predictions) > 1:
            self.max_k = len(predictions)
            self.agg_mode = f"pass@{len(predictions)}"
        else:
            self.max_k = 0
            self.agg_mode = "greedy"
        for pred in predictions:
            problem_name = pred["name"]
            self.predictions_by_problem[problem_name].append(pred)

    def get_problem_score(self, submissions) -> float:
        """
        For a given problem (list of submissions), compute the score as follows:
          - For each subtask, take the maximum score over all submissions.
          - Sum these maximum scores to get the problem score.
        """
        if not submissions:
            return 0.0
        subtasks = list(submissions[0]["test_case_results"].keys())
        subtask_scores = {subtask: 0 for subtask in subtasks}

        for submission in submissions:
            test_case_results = submission["test_case_results"]
            for subtask, result in test_case_results.items():
                subtask_scores[subtask] = max(subtask_scores[subtask], result["score"])
        return sum(subtask_scores.values()), subtask_scores

    def get_metrics(self):
        total_score = 0.0
        total_submissions = 0
        success_cnt = 0

        for _, submissions in self.predictions_by_problem.items():
            # aggregate problem score
            score, _ = self.get_problem_score(submissions)
            total_score += score

            # counts for total & successful submissions
            total_submissions += len(submissions)
            for sub in submissions:
                tc_results = sub.get("test_case_results")
                if tc_results and len(tc_results) > 0:
                    success_cnt += 1

        metrics = {self.agg_mode: {}}
        self.update_common_metrics(metrics[self.agg_mode])
        metrics[self.agg_mode]["Total Generations"] = total_submissions
        metrics[self.agg_mode]["Successfully Evaluated"] = success_cnt
        metrics[self.agg_mode]["Total Score"] = str(round(total_score))
        return metrics

    def evaluations_to_print(self):
        return [f"pass@{self.max_k}"]

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
