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
import shutil
import sys
from collections import Counter, defaultdict
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from typing import Any, List, Tuple

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics import read_predictions
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class ProcessTopAnswerConfig:
    """Top-level parameters for the script"""

    # Input_dir relative to which all the input_files are specified
    input_dir: str
    # Input files relative to input_dir which are used for majority voting
    # Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_dir/output-rs*.jsonl"
    input_files: Any

    # The script can be run in two modes:
    # 1. fill: use the best answer as the expected_answer to fill input_files
    # 2. extract: identify the best answer from input_files
    mode: str

    # Output directory is optional depending on whether the task is to fill the majority answer
    # or to just extract the best answer
    output_dir: str | None = None

    # which field to put the top scoring answer in
    fill_key: str | None = None

    # if True, will not change the fill_key if it's already filled with not None
    ignore_if_not_none: bool = False

    # if True, will use string match to fill is_correct key
    fill_is_correct: bool = True

    # if True, use the highest scoring answer from the RM as the majority answer
    use_highest_rm_score: bool = False

    # if True, will use the RM score for weighted majority voting
    use_majority_rm_score: bool = False

    # if provided, will fail if can't find this many files. Useful in scheduled
    # pipelines to ensure this step doesn't run if some of the expected files are missing
    require_num_files: int | None = None

    def __post_init__(self):
        """Building data_file from dataset/split if not provided directly."""
        if isinstance(self.input_files, str):
            if ',' in self.input_files:
                self.input_files = self.input_files.split(",")
            else:
                self.input_files = self.input_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_process_top_answer_config", node=ProcessTopAnswerConfig)


def map_to_output_path(file_path, input_dir, output_dir):
    """Map the input file path to the output file path"""
    # Convert all to Path objects
    file_path, input_dir, output_dir = Path(file_path), Path(input_dir), Path(output_dir)

    # Get the relative path from input_dir to the file
    relative_path = file_path.relative_to(input_dir)

    # Combine output_dir with the relative path
    output_path = output_dir / relative_path

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


class ProcessMode(Enum):
    FILL = "fill"
    EXTRACT = "extract"


class TopAnswerProcessor:
    def __init__(self, cfg: ProcessTopAnswerConfig):
        self.cfg = cfg
        self._validate_cfg()

    def _validate_cfg(self):
        """Validate the config"""
        cfg = self.cfg
        # Check if the mode is valid
        try:
            self.process_mode = ProcessMode(cfg.mode)
        except ValueError:
            raise ValueError(f"Invalid mode: {cfg.mode}")

        # For fill mode, output_dir is required
        if self.process_mode == ProcessMode.FILL:
            if cfg.output_dir is None:
                raise ValueError("output_dir is required when mode is fill")

        # Check whether the input_dir and output_dir are valid
        if not Path(cfg.input_dir).exists():
            raise ValueError(f"Input directory does not exist: {cfg.input_dir}")
        if cfg.output_dir is not None and not Path(cfg.output_dir).exists():
            LOG.info("Output directory does not exist: %s", cfg.output_dir)
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        """Setup input and output file handles"""
        cfg = self.cfg
        # Process the input files
        self.input_files = list(unroll_files(cfg.input_files, parent_dir=cfg.input_dir))
        self.input_file_handles = [open(file, "rt", encoding="utf-8") for file in self.input_files]

        if self.cfg.require_num_files is not None:
            if len(self.input_file_handles) != cfg.require_num_files:
                raise ValueError(f"Expected {cfg.require_num_files} files, found {len(self.input_file_handles)}")

        if self.process_mode == ProcessMode.FILL:
            # Create output files and their handles with the same relative paths as the input files
            output_files = [
                map_to_output_path(file, cfg.input_dir, cfg.output_dir)
                for file in unroll_files(cfg.input_files, parent_dir=cfg.input_dir)
            ]
            self.output_file_handles = [open(file, "wt", encoding="utf-8") for file in output_files]

        elif self.process_mode == ProcessMode.EXTRACT:
            if cfg.output_dir is None:
                cfg.output_dir = cfg.input_dir

            # A single output file "output-agg.jsonl" is created where the top-scoring answer is
            # considered as predicted_answer for each problem
            self.output_file_handles = [open(Path(cfg.output_dir) / f"output-agg.jsonl", "wt", encoding="utf-8")]

        # Fill mode is used to indicate the mode of selecting the top-scoring answer
        self.fill_mode = "majority"
        if self.cfg.use_majority_rm_score:
            self.fill_mode = "majority_rm"
        elif self.cfg.use_highest_rm_score:
            self.fill_mode = "highest_rm"

        # Setting up the fill_key in case it's not provided
        if cfg.fill_key is None:
            if self.process_mode == ProcessMode.FILL:
                # During fill mode, the top-scoring answer is considered as ground truth
                cfg.fill_key = "expected_answer"
            elif self.process_mode == ProcessMode.EXTRACT:
                # During extract mode, the top-scoring answer is considered as predicted_answer
                cfg.fill_key = "predicted_answer"

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close all the input and output file handles"""
        for file_handle in self.input_file_handles:
            file_handle.close()
        for file_handle in self.output_file_handles:
            file_handle.close()

    def process(self):
        """Process the predictions and write the results to the output file(s)"""
        all_predictions, new_answers = self._read_predictions()
        self._write_results(all_predictions, new_answers)

    def _read_predictions(self) -> Tuple[List, List]:
        """Read the predictions from the input file(s)"""
        cfg = self.cfg
        new_answers = []
        all_predictions = []
        for idx, predictions in enumerate(tqdm(zip_longest(*self.input_file_handles, fillvalue="{}"))):
            data = read_predictions(predictions, idx, self.input_file_handles)

            # Store the metadata about correctness and judgement for each answer
            # Useful when extracting the top answer
            answer_to_metadata = {}
            for elem in data:
                if 'predicted_answer' not in elem:
                    elem['predicted_answer'] = extract_answer(elem['generation'])
                if elem['predicted_answer'] is not None:
                    answer_to_metadata[elem['predicted_answer']] = [
                        elem.get('is_correct', None),
                        elem.get('judgement', None),
                    ]

            all_predictions.append(data)

            if cfg.use_majority_rm_score or cfg.use_highest_rm_score:
                valid_answers_and_scores = [
                    (elem['predicted_answer'], elem['reward_model_score'])
                    for elem in data
                    if elem['predicted_answer'] is not None
                ]
                new_answers.append(("no_valid_answer_found", 0, None, None))
                if len(valid_answers_and_scores) == 0:
                    continue

                # Calculate the score for each answer
                # TODO: This dictionary is just using surface form matching. Need to adapt for semantic matching.
                answer_scores = defaultdict(float)
                if cfg.use_majority_rm_score:
                    for answer, score in valid_answers_and_scores:
                        answer_scores[answer] += score
                else:
                    # Choose the max score for each answer
                    for answer, score in valid_answers_and_scores:
                        answer_scores[answer] = max(answer_scores[answer], score)

                # Answer is the top-scoring reward model score
                rm_answer, rm_score = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)[0]
                new_answers[-1] = [rm_answer, rm_score] + answer_to_metadata[rm_answer]
            else:
                # Perform majority voting
                # TODO: currently majority does not take into account equivalent answers written in a different way
                valid_answers = [elem['predicted_answer'] for elem in data if elem['predicted_answer'] is not None]
                new_answers.append(("no_valid_answer_found", (0, len(self.input_file_handles)), None, None))
                if len(valid_answers) == 0:
                    continue
                majority_answer, num_votes = Counter(valid_answers).most_common(1)[0]
                new_answers[-1] = [majority_answer, (num_votes, len(self.input_file_handles))] + answer_to_metadata[
                    majority_answer
                ]

        return all_predictions, new_answers

    def _write_results(self, all_predictions: List, new_answers: List):
        """Write the results to the output file(s)"""
        if self.process_mode == ProcessMode.FILL:
            self._write_results_fill(all_predictions, new_answers)
        elif self.process_mode == ProcessMode.EXTRACT:
            self._write_results_extract(all_predictions, new_answers)

    def _write_results_fill(self, all_predictions: List, new_answers: List):
        """Fill the expected_answer with the top answer"""
        cfg = self.cfg
        total_problems_changed, total_solutions_changed = 0, 0
        for idx, predictions in enumerate(all_predictions):
            changed = False
            for fidx, handle in enumerate(self.output_file_handles):
                if cfg.ignore_if_not_none and predictions[fidx].get(cfg.fill_key):
                    handle.write(json.dumps(predictions[fidx]) + "\n")
                    continue

                if predictions[fidx].get(cfg.fill_key) != new_answers[idx][0]:
                    total_solutions_changed += 1
                    changed = True

                # Add fill mode to the predictions
                predictions[fidx]["fill_mode"] = self.fill_mode
                # Fill the expected_answer with the top-scoring answer
                predictions[fidx][cfg.fill_key] = new_answers[idx][0]

                if cfg.use_majority_rm_score or cfg.use_highest_rm_score:
                    predictions[fidx]["answer_rm_score"] = new_answers[idx][1]
                else:
                    predictions[fidx]["majority_votes"], predictions[fidx]["total_votes"] = new_answers[idx][1]

                if cfg.fill_is_correct:
                    predictions[fidx]["is_correct"] = (
                        predictions[fidx]["predicted_answer"] == predictions[fidx]["expected_answer"]
                    )
                else:
                    predictions[fidx].pop("is_correct", None)

                handle.write(json.dumps(predictions[fidx]) + "\n")

            if changed:
                total_problems_changed += 1

        LOG.info(
            "Total problems changed: %d, total solutions changed: %d",
            total_problems_changed,
            total_solutions_changed,
        )

    def _write_results_extract(self, all_predictions: List, new_answers: List):
        """Write the top answer from the predictions to the output file"""
        best_answer_file_handle = self.output_file_handles[0]
        with open(self.input_files[0], "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                # Add fill mode to the predictions
                data["fill_mode"] = self.fill_mode

                data["predicted_answer"] = new_answers[idx][0]
                if new_answers[idx][2] is not None:
                    data["is_correct"] = new_answers[idx][2]
                if new_answers[idx][3] is not None:
                    data["judgement"] = new_answers[idx][3]
                best_answer_file_handle.write(json.dumps(data) + "\n")


@hydra.main(version_base=None, config_name="base_process_top_answer_config")
def process_top_answer(cfg: ProcessTopAnswerConfig):
    cfg = ProcessTopAnswerConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    with TopAnswerProcessor(cfg) as processor:
        processor.process()


HELP_MESSAGE = get_help_message(ProcessTopAnswerConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        process_top_answer()
