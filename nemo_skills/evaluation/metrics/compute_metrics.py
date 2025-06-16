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
    def __init__(
        self,
        benchmark,
        data_dir=None,
        cluster_config=None,
        extra_datasets=None,
        extra_datasets_type=None,
        max_samples=-1,
        metric_type=None,
    ):
        self.max_samples = max_samples
        self.metric_type = metric_type

        benchmark_module, _, _ = get_dataset_module(
            benchmark,
            data_dir=data_dir,
            cluster_config=cluster_config,
            extra_datasets=extra_datasets,
            extra_datasets_type=extra_datasets_type,
        )
        if self.metric_type is None:
            self.metric_type = benchmark_module.METRICS_TYPE

        # Dictionary to store metrics calculators for different subsets
        self.calculators = {}

    def get_metrics_calculator(self):
        metrics_calculator = get_metrics(self.metric_type)
        metrics_calculator.reset()
        return metrics_calculator

    def compute_metrics(self, input_files):
        """Computing metrics based on the provided input files."""
        # only calling setup on the main one
        self.calculators = {'all': self.get_metrics_calculator()}
        self.calculators['all'].setup(input_files)

        # sorting input files to ensure consistent order
        input_files = sorted(input_files)

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
                    self.calculators[data_subset] = self.get_metrics_calculator()
                self.calculators['all'].update(data)
                if data_subset != 'all':
                    self.calculators[data_subset].update(data)

        # collecting metrics from all calculators
        metrics = {}
        for data_subset, calculator in self.calculators.items():
            metrics[data_subset] = calculator.get_metrics()
            # if there is only a single prediction output.jsonl
            # we are renaming pass@1 to greedy to be consistent with ns eval logic
            if len(input_files) == 1 and input_files[0].endswith('output.jsonl'):
                if 'pass@1[1]' in metrics[data_subset]:
                    metrics[data_subset]['greedy'] = metrics[data_subset].pop('pass@1[1]')
                if 'pass@1' in metrics[data_subset]:
                    metrics[data_subset]['greedy'] = metrics[data_subset].pop('pass@1')
        return metrics

    def metrics_to_print(self):
        return self.calculators['all'].metrics_to_print()

    def evaluations_to_print(self):
        return self.calculators['all'].evaluations_to_print()
