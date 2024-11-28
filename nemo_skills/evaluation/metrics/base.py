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
import importlib
from contextlib import ExitStack
from itertools import zip_longest

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.metrics.utils import read_predictions
from nemo_skills.utils import unroll_files


# Base class for metrics computation
class BaseMetrics(abc.ABC):
    @abc.abstractmethod
    def fill_up_missing(self):
        pass

    @abc.abstractmethod
    def is_incomplete(self, elem):
        pass

    @abc.abstractmethod
    def update(self, predictions):
        pass

    @abc.abstractmethod
    def get_metrics(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def setup(self, input_files):
        pass

    def max_metrics_to_print(self):
        """No limit by default."""
        return None


class ComputeMetrics:
    def __init__(self, benchmark, extra_datasets=None, max_samples=-1):
        self.benchmark = benchmark

        # Setup metrics calculator
        benchmark_module, _ = get_dataset_module(benchmark, extra_datasets=extra_datasets)
        self.metrics_calculator = benchmark_module.METRICS_CLASS()

        self.max_samples = max_samples

    def compute_metrics(self, input_files, allow_incomplete=False):
        self.metrics_calculator.setup(input_files)
        self.metrics_calculator.reset()

        with ExitStack() as stack:
            file_handles = [
                stack.enter_context(open(file, "rt", encoding="utf-8")) for file in unroll_files(input_files)
            ]

            for idx, predictions in enumerate(zip_longest(*file_handles)):
                if idx == self.max_samples:
                    break
                data = read_predictions(predictions, self.metrics_calculator, allow_incomplete)
                self.metrics_calculator.update(data)

            metrics_dict = self.metrics_calculator.get_metrics()

        return metrics_dict

    def max_metrics_to_print(self):
        return self.metrics_calculator.max_metrics_to_print()
