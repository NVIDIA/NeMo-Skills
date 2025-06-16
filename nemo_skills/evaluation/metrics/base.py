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

import abc
from collections import Counter, defaultdict


# Base class for metrics computation
class BaseMetrics(abc.ABC):

    def __init__(self):
        self.reset()

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.eval_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}
            if self.avg_tokens > 0:
                metrics_dict[agg_mode]['avg_tokens'] = int(self.avg_tokens / self.total)
            for metric_key, metric_value in agg_metric_dict.items():
                if isinstance(metric_value, float):
                    # by default we will return all float metrics as percentages
                    metrics_dict[agg_mode][metric_key] = 100.0 * metric_value / self.total
                else:
                    metrics_dict[agg_mode][metric_key] = metric_value

        return metrics_dict

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """
        Returns a dictionary with all applicable ways to measure the correctness score for a given prediction.

        Examples:

        {'correct': True}
        {'symbolic_correct': True, 'judge_correct': False}
        {
            'prompt_correct_strict': 0,
            'instruction_correct_strict': 12,
            'prompt_correct_loose': 1,
            ...
        }

        Can return anything that we can take a maximum over.
        """
        raise NotImplementedError(
            "Needs be implemented in the subclass to use built-in _compute_pass_at_k and _compute_majority_at_k methods."
        )

    def update(self, predictions):
        self.total += 1
        if self.max_k > 0 and len(predictions) != self.max_k:
            raise ValueError(
                f"Expected {self.max_k} predictions, but got {len(predictions)}. "
                "This is likely due to a mismatch in the number of generations for different test examples."
            )
        if self.max_k == 0:
            self.max_k = len(predictions)
        self.avg_tokens += sum(
            pred['num_generated_tokens'] for pred in predictions if 'num_generated_tokens' in pred
        ) / len(predictions)

    def reset(self):
        self.total = 0
        self.max_k = 0
        self.avg_tokens = 0
        self.eval_dict = defaultdict(lambda: defaultdict(float))

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
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_majority_at_k` in case there are other metrics we want to log.

        This method is being called in a loop for each score_method, so only
        use it for metrics that depend on the correctness method.
        """

    def _update_metrics_for_majority(
        self,
        eval_dict: dict,
        k: int,
        predictions: list[dict],
        predicted_answers: list[str],
    ):
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        Unlike `_update_score_metrics_for_pass`, this method is called one time after the
        loop over all `score_method` in `_compute_pass_at_k`.

        It can be used for metrics that do not depend on the correctness method.
        """

    def _compute_majority_at_k(
        self, predictions: list[dict], predicted_answers: list[str], eval_dict: dict | None = None
    ):
        """
        Get majority@k metrics for a given set of prediction results.

        Args:
            predictions (list): List of generated predictions.
                Will call `_get_score_dict` to get scores for predictions.
            predicted_answers (list): List of the answers that we should use to compute majority.
            eval_dict (Optional[dict]): Dictionary to store aggregated metrics.
                By default will use self.eval_dict.
        """
        if eval_dict is None:
            eval_dict = self.eval_dict

        score_dicts = [self._get_score_dict(pred) for pred in predictions]

        for k in range(2, len(predictions) + 1):
            for score_method in score_dicts[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (pred_answer, correctness_dict[score_method])
                    for pred_answer, correctness_dict in zip(predicted_answers[:k], score_dicts[:k])
                    if pred_answer is not None
                ]

                # If no valid answers, it's incorrect
                if not valid_answers_and_results:
                    majority_score = 0
                    majority_answer = None
                else:
                    # sorting to ensure reproducible scores in case of ties in majority
                    valid_answers_and_results = sorted(valid_answers_and_results)
                    # Find the most common answer and its correctness
                    majority_answer, majority_score = Counter(valid_answers_and_results).most_common(1)[0][0]

                eval_dict[f"majority@{k}"][score_method] += majority_score

                # TODO: implement "avg_correct_tokens", "avg_incorrect_tokens" and "majority_ties" metrics

                # In case there are other metrics we need to update
                self._update_score_metrics_for_majority(
                    eval_dict=eval_dict,
                    k=k,
                    score_method=score_method,
                    score_dicts=score_dicts,
                    majority_score=majority_score,
                    majority_answer=majority_answer,
                    predictions=predictions,
                    predicted_answers=predicted_answers,
                )

            eval_dict[f"majority@{k}"]["no_answer"] += all(answer is None for answer in predicted_answers[:k])
            self._update_metrics_for_majority(
                eval_dict=eval_dict,
                k=k,
                predictions=predictions,
                predicted_answers=predicted_answers,
            )

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
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        This method is being called in a loop for each score_method, so only
        use it for metrics that depend on the correctness method.
        """

    def _update_metrics_for_pass(
        self,
        eval_dict: dict,
        k: int,
        predictions: list[dict],
        predicted_answers: list[str] | None,
    ):
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        Unlike `_update_score_metrics_for_pass`, this method is called one time after the
        loop over all `score_method` in `_compute_pass_at_k`.

        It can be used for metrics that do not depend on the correctness method.
        """

    def _compute_pass_at_k(
        self, predictions: list[dict], predicted_answers: list[str] | None = None, eval_dict: dict | None = None
    ):
        """
        Get pass@k metrics for a given set of prediction results.

        Args:
            predictions (list): List of generated predictions.
                Will call `_get_score_dict` to see which predictions are correct.
            predicted_answers (Optional[list]): List of the answers that will be used to compute no_answer metric.
            eval_dict (Optional[dict]): Dictionary to store aggregated metrics.
                By default will use self.eval_dict.
        """
        if eval_dict is None:
            eval_dict = self.eval_dict
        score_dicts = [self._get_score_dict(pred) for pred in predictions]

        for k in range(1, len(predictions) + 1):
            for score_method in score_dicts[0].keys():
                # TODO: implement "avg_correct_tokens", "avg_incorrect_tokens" metrics

                scores_list = [correctness_dict[score_method] for correctness_dict in score_dicts[:k]]
                pass_score = max(scores_list)
                eval_dict[f"pass@{k}"][score_method] += pass_score

                # pass@1[k] - mean of pass@1 across all generations
                eval_dict[f"pass@1[{k}]"][score_method] += sum(scores_list) / k

                self._update_score_metrics_for_pass(
                    eval_dict=eval_dict,
                    k=k,
                    score_method=score_method,
                    score_dicts=score_dicts,
                    pass_score=pass_score,
                    predictions=predictions,
                    predicted_answers=predicted_answers,
                )
            if predicted_answers is not None:
                no_answer_list = [pred_answer is None for pred_answer in predicted_answers[:k]]
                eval_dict[f"pass@{k}"]["no_answer"] += all(no_answer_list)
                eval_dict[f"pass@1[{k}]"]["no_answer"] += sum(no_answer_list) / k

            self._update_metrics_for_pass(
                eval_dict=eval_dict,
                k=k,
                predictions=predictions,
                predicted_answers=predicted_answers,
            )

    def setup(self, input_files):
        pass

    def metrics_to_print(self):
        """No limit by default."""
        return None

    def evaluations_to_print(self):
        """We will log all pass/pass@1[k] up to k, but only report the kth one."""
        return [f'pass@1[{self.max_k}]', f'majority@{self.max_k}', f'pass@{self.max_k}']


def as_percentage(metric_value):
    return f"{metric_value:.2f}%"


def as_int(metric_value):
    return f"{int(metric_value)}"


def as_float(metric_value):
    return f"{float(metric_value):.2f}"


def default_formatting(metric_value):
    """Assumes floats are percentage and rest without changes."""
    if isinstance(metric_value, float):
        return as_percentage(metric_value)
    return str(metric_value)
