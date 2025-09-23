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

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import tqdm

from nemo_skills.utils import unroll_files


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
        for input_file in tqdm.tqdm(unroll_files(input_files), desc="Processing files"):
            # assume that input_file is small enough to entirely fit in the memory
            input_data = []
            with open(input_file, "rt", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)

            with open(input_file, "rt", encoding="utf-8") as fin:
                # TODO we could possibly make this more efficient by allowing concurrent processing, but this is an okay base impl
                for file_line in tqdm.tqdm(fin, total=num_lines, desc=f"Evaluating {os.path.basename(input_file)}"):
                    line_dict = json.loads(file_line)
                    line_dict = await self.eval_single(line_dict)
                    input_data.append(line_dict)

            with open(input_file, "wt", encoding="utf-8", buffering=1) as fout:
                for line_dict in input_data:
                    fout.write(json.dumps(line_dict) + "\n")

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
