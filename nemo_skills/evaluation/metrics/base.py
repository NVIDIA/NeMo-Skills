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
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total, "avg_tokens": int(self.avg_tokens / self.total)}
            for metric_key, metric_value in agg_metric_dict.items():
                if isinstance(metric_value, float):
                    # by default we will return all float metrics as percentages
                    metrics_dict[agg_mode][metric_key] = 100.0 * metric_value / self.total
                else:
                    metrics_dict[agg_mode][metric_key] = metric_value

        return metrics_dict

    def _get_correctness_dict(self, prediction: dict) -> dict[bool]:
        """
        Returns a dictionary with all applicable ways to measure if the prediction is correct.

        Examples:

        {'correct': True/False}
        {'symbolic_correct': True/False, 'judge_correct': True/False, ...}
        {
            'prompt_correct_strict': True/False,
            'instruction_correct_strict': True/False,
            'prompt_correct_loose': True/False,
            ...
        }
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
        self.agg_mode_dict = defaultdict(lambda: defaultdict(float))

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
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_majority_at_k` in case there are other metrics we want to log.

        This method is being called in a loop for each check_correctness_method, so only
        use it for metrics that depend on the correctness method.
        """

    def _update_metrics_for_majority(
        self,
        agg_mode_dict: dict,
        k: int,
        predictions: list[dict],
        predicted_answers: list[str],
    ):
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        Unlike `_update_correctness_metrics_for_pass`, this method is called one time after the
        loop over all `check_correctness_method` in `_compute_pass_at_k`.

        It can be used for metrics that do not depend on the correctness method.
        """

    def _compute_majority_at_k(
        self, predictions: list[dict], predicted_answers: list[str], agg_mode_dict: dict | None = None
    ):
        """
        Get majority@k metrics for a given set of prediction results.

        Args:
            predictions (list): List of generated predictions.
                Will call `_get_correctness_dict` to see which predictions are correct.
            predicted_answers (list): List of the answers that we should use to compute majority.
            agg_mode_dict (Optional[dict]): Dictionary to store aggregated metrics.
                By default will use self.agg_mode_dict.
        """
        agg_mode_dict = agg_mode_dict or self.agg_mode_dict

        correctness_dicts = [self._get_correctness_dict(pred) for pred in predictions]

        for k in range(2, len(predictions) + 1):
            for check_correctness_method in correctness_dicts[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (pred_answer, correctness_dict[check_correctness_method])
                    for pred_answer, correctness_dict in zip(predicted_answers[:k], correctness_dicts[:k])
                    if pred_answer is not None
                ]

                # If no valid answers, it's incorrect
                if not valid_answers_and_results:
                    is_correct = False
                    majority_answer = None
                else:
                    # Find the most common answer and its correctness
                    majority_answer, is_correct = Counter(valid_answers_and_results).most_common(1)[0][0]

                agg_mode_dict[f"majority@{k}"][check_correctness_method] += is_correct

                # by default logging "correct", "no_answer", "avg_correct_tokens", "avg_incorrect_tokens" and "majority_ties"
                # TODO: implement above metrics

                # In case there are other metrics we need to update
                self._update_correctness_metrics_for_majority(
                    agg_mode_dict=agg_mode_dict,
                    k=k,
                    check_correctness_method=check_correctness_method,
                    is_correct=is_correct,
                    predictions=predictions,
                    predicted_answers=predicted_answers,
                    correctness_dicts=correctness_dicts,
                    majority_answer=majority_answer,
                )

            agg_mode_dict[f"majority@{k}"]["no_answer"] += all(answer is None for answer in predicted_answers[:k])
            self._update_metrics_for_majority(
                agg_mode_dict=agg_mode_dict,
                k=k,
                predictions=predictions,
                predicted_answers=predicted_answers,
            )

    def _update_correctness_metrics_for_pass(
        self,
        agg_mode_dict: dict,
        k: int,
        check_correctness_method: str,
        is_correct: bool,
        predictions: list[dict],
        correctness_dicts: list[dict],
    ):
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        This method is being called in a loop for each check_correctness_method, so only
        use it for metrics that depend on the correctness method.
        """

    def _update_metrics_for_pass(
        self,
        agg_mode_dict: dict,
        k: int,
        predictions: list[dict],
    ):
        """
        Update the metrics dictionary with additional statistics.

        Called by `_compute_pass_at_k` in case there are other metrics we want to log.

        Unlike `_update_correctness_metrics_for_pass`, this method is called one time after the
        loop over all `check_correctness_method` in `_compute_pass_at_k`.

        It can be used for metrics that do not depend on the correctness method.
        """

    def _compute_pass_at_k(
        self, predictions: list[dict], predicted_answers: list[str] | None = None, agg_mode_dict: dict | None = None
    ):
        """
        Get pass@k metrics for a given set of prediction results.

        Args:
            predictions (list): List of generated predictions.
                Will call `_get_correctness_dict` to see which predictions are correct.
            predicted_answers (Optional[list]): List of the answers that will be used to compute no_answer metric.
            agg_mode_dict (Optional[dict]): Dictionary to store aggregated metrics.
                By default will use self.agg_mode_dict.
        """
        agg_mode_dict = agg_mode_dict or self.agg_mode_dict
        correctness_dicts = [self._get_correctness_dict(pred) for pred in predictions]

        for k in range(1, len(predictions) + 1):
            for check_correctness_method in correctness_dicts[0].keys():
                # by default logging "correct", "avg_correct_tokens", "avg_incorrect_tokens"
                # TODO: implement above metrics

                is_correct_list = [
                    correctness_dict[check_correctness_method] for correctness_dict in correctness_dicts[:k]
                ]
                is_correct = any(is_correct_list)
                agg_mode_dict[f"pass@{k}"][check_correctness_method] += is_correct

                # pass@1[k] - mean of pass@1 across all generations
                agg_mode_dict[f"pass@1[{k}]"][check_correctness_method] += sum(is_correct_list) / k

                self._update_correctness_metrics_for_pass(
                    agg_mode_dict=agg_mode_dict,
                    k=k,
                    check_correctness_method=check_correctness_method,
                    is_correct=is_correct,
                    predictions=predictions,
                    correctness_dicts=correctness_dicts,
                )
            if predicted_answers is not None:
                no_answer_list = [pred_answer is None for pred_answer in predicted_answers[:k]]
                agg_mode_dict[f"pass@{k}"]["no_answer"] += all(no_answer_list)
                agg_mode_dict[f"pass@1[{k}]"]["no_answer"] += sum(no_answer_list) / k

            self._update_metrics_for_pass(
                agg_mode_dict=agg_mode_dict,
                k=k,
                predictions=predictions,
            )

    def setup(self, input_files):
        pass

    def metrics_to_print(self):
        """No limit by default."""
        return None

    def aggregations_to_print(self):
        """No limit by default."""
        return None


def as_percentage(metric_value):
    return f"{metric_value:.2f}%"


def as_int(metric_value):
    return f"{int(metric_value)}"


def default_formatting(metric_value):
    """Assumes floats are percentage and rest without changes."""
    if isinstance(metric_value, float):
        return as_percentage(metric_value)
    return str(metric_value)
