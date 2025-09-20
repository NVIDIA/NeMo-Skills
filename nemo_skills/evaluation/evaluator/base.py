# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with configuration."""
        self.config = config

    @abstractmethod
    async def eval_full(self, input_files: List[str], **kwargs) -> None:
        """
        Evaluate full dataset in batch mode.

        Args:
            input_files: List of input files to evaluate
            **kwargs: Additional evaluation parameters
        """
        pass

    async def eval_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single data point during generation (optional).

        Args:
            data_point: Single data point with generation results

        Returns:
            Dict with evaluation results to merge into data_point

        Raises:
            NotImplementedError: If single evaluation is not supported
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support single evaluation during generation")

    def supports_single_eval(self) -> bool:
        """Check if this evaluator supports single evaluation during generation."""
        return self.__class__.eval_single is not BaseEvaluator.eval_single
