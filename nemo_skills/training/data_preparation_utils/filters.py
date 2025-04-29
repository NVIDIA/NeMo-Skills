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

import json
import logging
import os
import re
import warnings
from itertools import chain
from math import isclose
from pathlib import Path
from typing import List

import tqdm
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from tqdm.contrib.concurrent import process_map

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.prompt.utils import load_config
from nemo_skills.training.data_preparation_utils.arithmetic_utils import (
    extract_expressions,
    merge_solution_steps,
    solve_expression,
)

LOG = logging.getLogger(__file__)

PREFIX_SOLN = "My solution:\n"
PATTERN_ANS = re.compile(r"\\boxed\{([^}]*)\}")
PATTERN_PYTHON_CODE = re.compile("```[pP]ython")


class BaseFilter(BaseParallelProcessor):
    def __init__(self, **kwargs):
        if 'in_memory_chunksize' not in kwargs:
            kwargs['in_memory_chunksize'] = 250000
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 2500
        if 'max_workers' not in kwargs:
            kwargs['max_workers'] = max(100, os.cpu_count())
        super().__init__(**kwargs)

    def finalize(self, metrics: List):
        LOG.info("Number of entries after processing: %d", self.number_of_entries)

        if not metrics:
            return

        if 'num_removed' in metrics[0]:
            num_removed_entries = sum(metric.get('num_removed', 0) for metric in metrics)
            LOG.info("Number of removed entries: %d", num_removed_entries)

        if 'num_modified' in metrics[0]:
            num_modified_entries = sum(metric.get('num_modified', 0) for metric in metrics)
            LOG.info("Number of modified entries: %d", num_modified_entries)

    def _chunk_manifest(self):
        """Small modification of original function to print progress."""
        manifest_chunk = []
        for idx, data_entry in enumerate(self.read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                LOG.info("Processing chunk %d", idx)
                yield manifest_chunk
                manifest_chunk = []
        if len(manifest_chunk) > 0:
            LOG.info("Processing last chunk")
            yield manifest_chunk


class DropIfRegexMatch(BaseFilter):
    """Drops data if text matches a regex pattern."""

    def __init__(
        self,
        regex_patterns: List[str],
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_patterns = regex_patterns
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        for regex_pattern in self.regex_patterns:
            if re.search(re.escape(regex_pattern), data_entry[self.text_key]):
                return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_reomoved=0))]


class DropIfRegexNotMatch(BaseFilter):
    """Drops data if text doesn't match at least one of the regex patterns."""

    def __init__(
        self,
        regex_patterns: List[str],
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_patterns = regex_patterns
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        for regex_pattern in self.regex_patterns:
            if not re.search(re.escape(regex_pattern), data_entry[self.text_key]):
                return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_reomoved=0))]


class DropIfEqual(BaseFilter):
    """Drops data if entry matches provided value."""

    def __init__(
        self,
        values,
        key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values = values
        self.key = key

    def process_dataset_entry(self, data_entry) -> List:
        for value in self.values:
            if data_entry[self.key] == value:
                return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_reomoved=0))]


class DropMultiBoxed(BaseFilter):
    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if len(PATTERN_ANS.findall(data_entry[self.solution_key])) > 1:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class DropIncorrectCodeBlocks(BaseFilter):
    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if len(PATTERN_PYTHON_CODE.findall(data_entry[self.solution_key])) != 1:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]
        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class AddCodeExecutionsCounts(BaseFilter):
    def __init__(
            self, solution_key: str = "generation",
            ce_counter_key: str = "total_code_executions",
            **kwargs
        ):
        super().__init__(**kwargs)
        self.solution_key = solution_key
        self.ce_counter_key = ce_counter_key

    def process_dataset_entry(self, data_entry) -> List:
        pattern = r"Remaining code executions: (\d+)."
        allowed_ce = re.search(pattern, data_entry[self.solution_key])
        counts = 0
        if not allowed_ce:
            if "You have run out of code executions!" in data_entry[self.solution_key]:
                counts = 1
        else:
            counts = int(allowed_ce.group(1))

        data_entry[self.ce_counter_key] = counts
        return [DataEntry(data=data_entry, metrics=dict(num_modified=1))]


class DropIncorrectArithmetic(BaseFilter):
    def __init__(self, solution_key: str = "generation", tolerance=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key
        self.tolerance = tolerance

    def process_dataset_entry(self, data_entry: str) -> List:
        for expression, _ in extract_expressions(data_entry[self.solution_key]):
            parts = expression.split("=")
            if len(parts) < 2:
                continue

            expr, ans = parts[0], parts[-1]

            try:
                solution_steps = solve_expression(expr)
                # ignore eval warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=SyntaxWarning)
                    if not isclose(eval(solution_steps[-1]), eval(ans), rel_tol=self.tolerance):
                        return [DataEntry(data=None, metrics=dict(num_removed=1))]
            except KeyboardInterrupt:
                raise
            except:
                pass

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class MajorityFilter(BaseFilter):
    def __init__(
        self,
        min_majority_votes: int = 0,
        min_majority_percentage: int = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_majority_votes = min_majority_votes
        self.min_majority_percentage = min_majority_percentage

    def process_dataset_entry(self, data_entry) -> List:
        majority_votes = data_entry.get("majority_votes", None)
        total_votes = data_entry.get("total_votes", None)
        if majority_votes is None or total_votes is None:
            return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]
        if majority_votes < self.min_majority_votes or majority_votes < total_votes * self.min_majority_percentage:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class RemoveContaminated(BaseFilter):
    def __init__(self, contamination_file, check_key="problem", **kwargs):
        if contamination_file is None:
            raise ValueError("Specify 'contamination_file' to filter contaminated questions")
        super().__init__(**kwargs)
        self.check_key = check_key
        self.contaminated_entries = set()
        with open(contamination_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                if data["contaminated"]:
                    self.contaminated_entries.add(data[self.check_key])

    def process_dataset_entry(self, data_entry) -> List:
        if data_entry[self.check_key] in self.contaminated_entries:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class RemoveLenOutliers(BaseFilter):
    """Remove instance based on minimum and maximum lengths for a given property."""

    def __init__(
        self,
        property_key: str = "generation",
        min_length: int = 0,
        max_length: int = None,
        hf_model_name: str = None,
        use_chars_for_min_length: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.property_key = property_key
        self.max_length = max_length
        self.min_length = min_length
        self.use_chars_for_min_length = use_chars_for_min_length

        if hf_model_name is None:
            raise ValueError("Specify 'hf_model_name' for length-based filtering")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    def process_dataset_entry(self, data_entry):
        property_val = data_entry[self.property_key]
        property_len = len(self.tokenizer.encode(property_val, add_special_tokens=False))

        if self.use_chars_for_min_length:
            if len(property_val) < self.min_length:
                return [DataEntry(data=None, metrics=dict(num_removed=1))]
        else:
            if property_len < self.min_length:
                return [DataEntry(data=None, metrics=dict(num_removed=1))]

        if property_len > self.max_length:
            return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_removed=0))]


class TrimPrefix(BaseFilter):
    """Remove common prefix from solutions."""

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        if data_entry[self.solution_key].startswith(PREFIX_SOLN):
            data_entry[self.solution_key] = data_entry[self.solution_key][len(PREFIX_SOLN) :]
            return [DataEntry(data=data_entry, metrics=dict(num_modified=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_modified=0))]


class TrimSolutions(BaseFilter):
    """Trimming solutions till the last line with the answer in \\boxed{}."""

    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry) -> List:
        # extracting full boxed answer first
        predicted_answer = extract_answer(data_entry[self.solution_key])
        
        original_solution = data_entry[self.solution_key]
        if predicted_answer is not None:
            predicted_answer = "\\boxed{" + predicted_answer + "}"
            before, after = data_entry[self.solution_key].rsplit(predicted_answer, 1)
            data_entry[self.solution_key] = before + predicted_answer + after.split("\n")[0]
        is_modified = original_solution != data_entry[self.solution_key]

        return [DataEntry(data=data_entry, metrics=dict(num_modified=int(is_modified)))]


class SplitArithmetic(BaseFilter):
    def __init__(self, solution_key: str = "generation", **kwargs):
        super().__init__(**kwargs)
        self.solution_key = solution_key

    def process_dataset_entry(self, data_entry: str) -> List:
        """
        Extends short arithmetic expressions solutions to step-by-step ones
        For example `1 + 2 + 3 + 4 = 10` -> `1 + 2 + 3 + 4 = 3 + 3 + 4 = 6 + 4 = 10`.
        """
        text = data_entry[self.solution_key]
        new_text = []
        last_end = 0

        for expression, start in extract_expressions(text):
            end = start + len(expression)
            parts = expression.split("=")

            if len(parts) != 2:
                new_text.append(text[last_end:end])
                last_end = end
                continue
            expr, ans = parts

            try:
                solution_steps = solve_expression(expr)
            except:
                new_text.append(text[last_end:end])
                last_end = end
                continue

            solution = merge_solution_steps(solution_steps)

            try:
                # ignore eval warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=SyntaxWarning)
                    if eval(solution_steps[-1]) == eval(ans):
                        new_text.append(text[last_end:start] + solution)
                    else:
                        new_text.append(text[last_end:end])

                last_end = end
            except KeyboardInterrupt:
                raise
            except:
                new_text.append(text[last_end:end])
                last_end = end

        new_text.append(text[last_end:])
        data_entry[self.solution_key] = "".join(new_text)
        is_modified = text != data_entry[self.solution_key]

        return [DataEntry(data=data_entry, metrics=dict(num_modified=int(is_modified)))]


class CodeTextFilter(BaseParallelProcessor):
    def __init__(self, filter_type, prompt_template, solution_key='generation', **kwargs):
        if 'in_memory_chunksize' not in kwargs:
            kwargs['in_memory_chunksize'] = 100000000
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 100000
        super().__init__(**kwargs)
        self.prompt_template = prompt_template
        self.text_filter_type = filter_type
        self.solution_key = solution_key

    def process_dataset_entry(self, grouped_samples: List, code_begin_token: str):
        code_solns = []
        text_solns = []
        for sample in grouped_samples:
            if code_begin_token in sample[self.solution_key]:
                code_solns.append(sample)
            else:
                text_solns.append(sample)

        filtered_predictions = []
        if self.text_filter_type is None:
            filtered_predictions.extend(code_solns)
            filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'all':
            filtered_predictions.extend(code_solns)
        elif self.text_filter_type == 'majority_code':
            filtered_predictions.extend(code_solns)
            if len(code_solns) <= len(grouped_samples) // 2:
                filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'majority_text':
            if len(code_solns) > len(grouped_samples) // 2:
                filtered_predictions.extend(code_solns)
            else:
                filtered_predictions.extend(text_solns)
        elif self.text_filter_type == 'any_code':
            if code_solns:
                filtered_predictions.extend(code_solns)
            else:
                filtered_predictions.extend(text_solns)
        else:
            raise NotImplementedError(f"Filtering method {self.text_filter_type} not implemented")
        num_removed = len(grouped_samples) - len(filtered_predictions)

        return [DataEntry(data=filtered_predictions, metrics=dict(num_removed=num_removed))]

    def process(self):
        self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        metrics = []
        prompt = load_config(self.prompt_template, Path(__file__).absolute().parents[2] / 'prompt' / 'template')
        code_begin_token = prompt.config.template.code_begin

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for manifest_chunk in self._chunk_manifest():
                # this will unroll all inner lists
                data = chain(
                    *process_map(
                        self.process_dataset_entry,
                        manifest_chunk,
                        code_begin_token,
                        max_workers=self.max_workers,
                        chunksize=self.chunksize,
                    )
                )
                for data_entry in tqdm.tqdm(data):
                    metrics.append(data_entry.metrics)
                    if data_entry.data is None:
                        continue
                    json.dump(data_entry.data, fout, ensure_ascii=False)
                    self.number_of_entries += 1
                    fout.write("\n")

        self.finalize(metrics)

    def finalize(self, metrics: List):
        LOG.info("Number of entries after processing: %d", self.number_of_entries)

        if not metrics:
            return

        if 'num_removed' in metrics[0]:
            num_removed_entries = sum(metric.get('num_removed', 0) for metric in metrics)
            LOG.info("Number of removed entries: %d", num_removed_entries)
