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
from collections import Counter


# Base class for metrics computation
class BaseMetrics(abc.ABC):
    @abc.abstractmethod
    def update(self, predictions):
        pass

    @abc.abstractmethod
    def get_metrics(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def get_prediction_results(self, prediction):
        pass

    def get_majority_at_k(
        self, agg_mode_dict, predicted_answers, pred_keys=None, predictions=None, prediction_results=None
    ):
        """
        Get majority@k metrics for a given set of prediction results.

        Args:
            agg_mode_dict (dict): Dictionary to store aggregated metrics.
            predicted_answers (list): List of generated predictions.
            pred_keys (Optional[list]): List of keys to aggregate over. If not provided, it will use all keys in the prediction results.
            predictions (Optional[list]): List of generated predictions. These will go through `get_prediction_results` to get the prediction results.
            prediction_results (Optional[list]): List of prediction results. If provided, it will ignore the `predictions` argument.
        """
        if prediction_results is None:
            assert predictions is not None, "Either predictions or prediction_results must be provided"
            prediction_results = [self.get_prediction_results(pred) for pred in predictions]

        if pred_keys is None:
            pred_keys = prediction_results[0].keys()

        for k in range(len(prediction_results), 1, -1):
            for pred_field in prediction_results[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (pred, result[pred_field])
                    for pred, result in zip(predicted_answers[:k], prediction_results[:k])
                    if pred is not None
                ]

                # If no valid answers, mark as incorrect
                if not valid_answers_and_results:
                    agg_mode_dict[f"majority@{k}"][pred_field] += False
                    continue

                # Find the most common answer and its correctness
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]

                # Update the metric
                agg_mode_dict[f"majority@{k}"][pred_field] += majority_result[1]

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

        if len(prediction_results) == 1:
            # Single decoding
            for pred_field in pred_keys:
                agg_mode_dict["greedy"][pred_field] += prediction_results[0][pred_field]
        else:
            for k in range(len(prediction_results), 0, -1):
                # Custom pass@k implementation
                if pass_at_k_fn is not None:
                    pass_at_k_fn(agg_mode_dict[f"pass@{k}"], prediction_results[:k])

                for pred_field in pred_keys:
                    # Regular pass@k
                    if pass_at_k_fn is None:
                        agg_mode_dict[f"pass@{k}"][pred_field] += any(
                            [elem[pred_field] for elem in prediction_results[:k]]
                        )

                    # Pass@1[k] - mean of pass@1 across all generations
                    agg_mode_dict[f"pass@1[{k}]"][pred_field] += (
                        sum([elem[pred_field] for elem in prediction_results[:k]]) / k
                    )

    def setup(self, input_files):
        pass

    def max_metrics_to_print(self):
        """No limit by default."""
        return None

    def max_aggregations_to_print(self):
        """No limit by default."""
        return None
