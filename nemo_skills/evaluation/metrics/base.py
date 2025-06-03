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

    @abc.abstractmethod
    def get_metrics(self):
        pass

    @abc.abstractmethod
    def get_prediction_results(self, prediction):
        """
        Extract and compute evaluation metrics from a single prediction.

        This method transforms a raw prediction dictionary into a standardized results dictionary
        containing boolean or numeric metrics that can be aggregated across multiple predictions.
        The returned dictionary keys serve as metric names for pass@k, majority@k, and other
        aggregation strategies.

        E.g. could return {'correct': True} which will then be aggregated across all predictions and printed as

        evaluation_mode | correct
        pass@1[64]      | 2.56%
        majority@64     | 5.92%
        pass@64         | 8.62%

        or could return {'correct_sympy': True, 'correct_judge': False} which will then be aggregated separately
        for each key, e.g.:

        evaluation_mode | correct_sympy | correct_judge
        pass@1[64]      | 2.56%         | 4.23%
        majority@64     | 5.92%         | 8.45%
        pass@64         | 8.62%         | 16.78%

        """
        pass

    def get_prediction_statistics(self, prediction):
        """
        Get extra statistics for a single prediction.

        These are similar to the results returned by `get_prediction_results`, but are not used to check correctness.
        E.g. to compute majority@k we will use the values returned by `get_prediction_results`, and then call
        `get_prediction_statistics` to log additional information about just the final majority prediction.

        E.g. could return {'avg_tokens': 1042, 'no_answer': False} which will then be aggregated across all predictions
        and printed as:
        evaluation_mode | avg_tokens | no_answer
        pass@1[64]      | 2539       | 12.34%
        majority@64     | 1987       | 1.45%
        pass@64         | 1765       | 0.00%

        By default any non-float types (e.g. integers) are printed as is, and any floats are printed as percentages.
        """
        if 'num_generated_tokens' in prediction:
            return {'avg_tokens': prediction['num_generated_tokens']}
        return {}

    def update(self, predictions):
        self.total += 1
        self.max_k = max(self.max_k, len(predictions))

    def reset(self):
        self.total = 0
        self.max_k = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def get_majority_at_k(
        self, predictions: list[dict], predicted_answers: list[str], agg_mode_dict: dict | None = None
    ):
        """
        Get majority@k metrics for a given set of prediction results.

        Args:
            predicted_answers (list): List of the answers that we should use to compute majority.
            predictions (list): List of generated predictions.
                Will call `get_prediction_results` to get the prediction results for each element.
            agg_mode_dict (Optional[dict]): Dictionary to store aggregated metrics.
                By default will use self.agg_mode_dict.
        """
        agg_mode_dict = agg_mode_dict or self.agg_mode_dict
        prediction_results = [self.get_prediction_results(pred) for pred in predictions]

        for k in range(2, len(prediction_results) + 1):
            for pred_field in prediction_results[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (pred, result[pred_field])
                    for pred, result in zip(predicted_answers[:k], prediction_results[:k])
                    if pred is not None
                ]

                # If no valid answers, mark as incorrect
                if not valid_answers_and_results:
                    agg_mode_dict[f"majority@{k}"][pred_field]['correct'] += False
                    continue

                # Find the most common answer and its correctness
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]

                # Update the results
                agg_mode_dict[f"majority@{k}"][pred_field]['correct'] += majority_result[1]

                # Getting one of the answers to use for extra statistics
                majority_prediction = predictions[predicted_answers.index(majority_result[0])]

                # Add extra statistics for the majority answer
                agg_mode_dict[f"majority@{k}"][pred_field].update(self.get_prediction_statistics(majority_prediction))

    def get_pass_at_k(
        self, agg_mode_dict, pred_keys=None, predictions=None, pass_at_k_fn=None, prediction_results=None
    ):
        """
        Get pass@k metrics for a given set of prediction results.

        Args:
            agg_mode_dict (dict): Dictionary to store aggregated metrics.
            pred_keys (Optional[list]): List of keys to aggregate over. If not provided, it will use all keys in the prediction results.
            predictions (Optional[list]): List of generated predictions. These will go through `get_prediction_results` to get the prediction results.
            pass_at_k_fn (Optional[function]): Custom function to compute pass@k.
            prediction_results (Optional[list]): List of prediction results. If provided, it will ignore the `predictions` argument.
        """
        if prediction_results is None:
            assert predictions is not None, "Either predictions or prediction_results must be provided"
            prediction_results = [self.get_prediction_results(pred) for pred in predictions]

        if pred_keys is None:
            pred_keys = prediction_results[0].keys()
            # print(pred_keys)
        if len(prediction_results) == 1:
            # Single decoding
            for pred_field in pred_keys:
                agg_mode_dict["greedy"][pred_field]['correct'] += prediction_results[0][pred_field]
        else:
            for k in range(1, len(prediction_results) + 1):
                # Custom pass@k implementation
                if pass_at_k_fn is not None:
                    pass_at_k_fn(agg_mode_dict[f"pass@{k}"], prediction_results[:k])

                for pred_field in pred_keys:
                    # Regular pass@k
                    if pass_at_k_fn is None:
                        agg_mode_dict[f"pass@{k}"][pred_field]['correct'] += any(
                            [elem[pred_field] for elem in prediction_results[:k]]
                        )

                    # Pass@1[k] - mean of pass@1 across all generations
                    agg_mode_dict[f"pass@1[{k}]"][pred_field]['correct'] += (
                        sum([elem[pred_field] for elem in prediction_results[:k]]) / k
                    )

                    # Getting one of the answers to use for extra statistics
                    for prediction, prediction_result in zip(predictions[:k], prediction_results[:k]):
                        if prediction_result[pred_field]:
                            agg_mode_dict[f"pass@{k}"][pred_field].update(self.get_prediction_statistics(prediction))
                            break

    def setup(self, input_files):
        pass

    def metrics_to_print(self):
        """No limit by default."""
        return None

    def aggregations_to_print(self):
        """No limit by default."""
        return None
