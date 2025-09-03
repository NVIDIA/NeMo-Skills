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

import copy
from contextlib import ExitStack
from itertools import zip_longest

import numpy as np

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.evaluation.metrics.utils import read_predictions
from nemo_skills.utils import unroll_files


class ComputeMetrics:
    def __init__(
        self,
        benchmark,
        data_dir=None,
        cluster_config=None,
        extra_datasets=None,
        extra_datasets_type=None,
        max_samples=-1,
        metric_type=None,
        max_seq_len=None,
        monte_carlo_samples=50,  # Number of monte_carlo shuffles of generations
    ):
        self.max_samples = max_samples
        self.metric_type = metric_type
        self.max_seq_len = max_seq_len
        self.monte_carlo_samples = monte_carlo_samples
        if self.metric_type is None:
            benchmark_module, _, _ = get_dataset_module(
                benchmark,
                data_dir=data_dir,
                cluster_config=cluster_config,
                extra_datasets=extra_datasets,
                extra_datasets_type=extra_datasets_type,
            )
            self.metric_type = benchmark_module.METRICS_TYPE

        # Dictionary to store metrics calculators for different subsets.
        # Each subset maps to a list of calculators, one per monte_carlo sample.
        self.calculators = {}

    def get_metrics_calculator(self):
        metrics_calculator = get_metrics(self.metric_type)
        metrics_calculator.reset()
        return metrics_calculator

    def compute_metrics(self, input_files):
        """Computing metrics based on the provided input files."""
        self.calculators = {"_all_": [self.get_metrics_calculator() for _ in range(self.monte_carlo_samples)]}
        # Setup for only one calculator
        self.calculators["_all_"][0].setup(input_files)

        # sorting input files to ensure consistent order
        input_files = sorted(input_files)

        examples = []  # list of tuples: (subset_key, data_list)
        with ExitStack() as stack:
            file_handles = [
                stack.enter_context(open(file, "rt", encoding="utf-8")) for file in unroll_files(input_files)
            ]

            for idx, predictions in enumerate(zip_longest(*file_handles)):
                if idx == self.max_samples:
                    break
                data = read_predictions(predictions, idx, file_handles)
                if self.max_seq_len is not None:
                    # Mark prediction as incorrect if the number of generated tokens exceeds max_seq_len
                    for i in range(len(data)):
                        if int(data[i]["num_generated_tokens"]) <= self.max_seq_len:
                            continue
                        data[i] = self.calculators["_all_"][0].get_incorrect_sample(data[i])
                data_subset = data[0].get("subset_for_metrics", "_all_")
                if data_subset not in self.calculators:
                    self.calculators[data_subset] = [
                        self.get_metrics_calculator() for _ in range(self.monte_carlo_samples)
                    ]
                examples.append((data_subset, data))

        use_monte_carlo = self.metric_type not in ["arena"]  # Arena already uses monte_carlo for their metrics
        if use_monte_carlo:
            monte_carlo_metric_list = []
            for monte_carlo_idx in range(self.monte_carlo_samples):
                rng = np.random.RandomState(monte_carlo_idx)
                examples_monte_carlo = copy.deepcopy(examples)
                # Bootstrapping will sample with replacement, but it would underestimate pass@k
                # Shuffling is unbiased, but variance is underestimated.
                self.shuffle_per_example_generations(examples_monte_carlo, rng)
                bs_metrics = self.compute_metrics_for_examples(examples_monte_carlo, monte_carlo_idx)
                monte_carlo_metric_list.append(bs_metrics)
            metrics = _average_metrics_dicts(monte_carlo_metric_list, include_std_dev=True)
        else:
            metrics = self.compute_metrics_for_examples(examples, 0)
        return metrics

    def shuffle_per_example_generations(self, examples, rng):
        for subset_key, base_data in examples:
            rng.shuffle(base_data)

    def compute_metrics_for_examples(self, examples, monte_carlo_idx):
        for subset_key, base_data in examples:
            self.calculators["_all_"][monte_carlo_idx].update(base_data)
            if subset_key != "_all_":
                self.calculators[subset_key][monte_carlo_idx].update(base_data)

        # collecting metrics from all calculators
        metrics = {}
        for data_subset, calculators in self.calculators.items():
            metrics[data_subset] = calculators[monte_carlo_idx].get_metrics()
            # we are removing pass@1[avg-of-1] as it's the same as pass@1
            metrics[data_subset].pop("pass@1[avg-of-1]", None)
        return metrics

    def metrics_to_print(self):
        return self.calculators["_all_"][0].metrics_to_print()

    def evaluations_to_print(self):
        return self.calculators["_all_"][0].evaluations_to_print()


def _average_metrics_dicts(dicts_list, include_std_dev=False):
    """Recursively average a list of metrics dictionaries.

    Assumes all dictionaries share the same structure and numeric leaves.
    For list/tuple leaves, averages over corresponding indices.
    When include_std_dev is False, returns the mean of the metrics.
    When include_std_dev is True, returns a dict {"avg": mean, "std": std}.

    Raises a ValueError if the dictionaries have different keys (usually happens if the returned metric keys are not consistent between monte_carlo samples)
    """
    if not dicts_list:
        return {}

    # If leaf is numeric, compute mean directly
    first = dicts_list[0]
    if not isinstance(first, dict):
        if isinstance(first, (list, tuple)):
            result = []
            for i in range(len(first)):
                values = [d[i] for d in dicts_list]
                # If all values are the same, return the first one
                if all(v == values[0] for v in values):
                    result.append(values[0])
                else:
                    mean = sum(values) / len(values)
                    if include_std_dev:
                        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                        result.append({"avg": mean, "std": std})
                    else:
                        result.append(mean)
            return type(first)(result)  # Convert back to original type (list/tuple)
        else:
            # numeric or other leaf types
            values = [v for v in dicts_list]
            # If all values are the same, return the first one
            if all(v == values[0] for v in values):
                return values[0]
            mean = sum(values) / len(values)
            if include_std_dev:
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                return {"avg": mean, "std": std}
            return mean

    # Otherwise, recurse over keys
    averaged = {}
    keys = first.keys()
    for key in keys:
        values_for_key = [d[key] for d in dicts_list]
        averaged[key] = _average_metrics_dicts(values_for_key, include_std_dev)
    return averaged
