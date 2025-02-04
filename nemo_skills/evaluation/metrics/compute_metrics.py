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

from contextlib import ExitStack
from itertools import zip_longest

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.evaluation.metrics.utils import read_predictions
from nemo_skills.utils import unroll_files


class ComputeMetrics:
    def __init__(self, benchmark, extra_datasets=None, max_samples=-1, metric_type=None):
        self.max_samples = max_samples
        self.benchmark = benchmark
        self.extra_datasets = extra_datasets
        self.metric_type = metric_type
        # Dictionary to store metrics calculators for different subsets
        self.calculators = {}

    def get_metrics_calculator(self, benchmark, extra_datasets=None, metric_type=None):
        if metric_type is None:
            # Setup metrics calculator
            benchmark_module, _ = get_dataset_module(benchmark, extra_datasets=extra_datasets)
            metrics_calculator = get_metrics(benchmark_module.METRICS_TYPE)
        else:
            metrics_calculator = get_metrics(metric_type)
        metrics_calculator.reset()

        return metrics_calculator

    def compute_metrics(self, input_files):
        """Computing metrics based on the provided input files."""
        # only calling setup on the main one
        self.calculators = {'all': self.get_metrics_calculator(self.benchmark, self.extra_datasets, self.metric_type)}
        self.calculators['all'].setup(input_files)

        with ExitStack() as stack:
            file_handles = [
                stack.enter_context(open(file, "rt", encoding="utf-8")) for file in unroll_files(input_files)
            ]

            for idx, predictions in enumerate(zip_longest(*file_handles)):
                if idx == self.max_samples:
                    break
                data = read_predictions(predictions, idx, file_handles)
                # checking if we need to create a new metrics calculator
                data_subset = data[0].get('subset_for_metrics', 'all')
                if data_subset not in self.calculators:
                    self.calculators[data[0]['subset_for_metrics']] = self.get_metrics_calculator(
                        self.benchmark,
                        self.extra_datasets,
                        self.metric_type,
                    )
                self.calculators['all'].update(data)
                if data_subset != 'all':
                    self.calculators[data_subset].update(data)

        return {data_subset: calculator.get_metrics() for data_subset, calculator in self.calculators.items()}

    def max_metrics_to_print(self):
        return self.calculators['all'].max_metrics_to_print()

    def max_aggregations_to_print(self):
        return self.calculators['all'].max_aggregations_to_print()
