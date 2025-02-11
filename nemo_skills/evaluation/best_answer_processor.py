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
from itertools import zip_longest
from pathlib import Path
from typing import Any

import hydra
from omegaconf import MISSING
from tqdm import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics import read_predictions
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class FillMajorityAnswerConfig:
    """Top-level parameters for the script"""

    # Input_dir relative to which all the input_files are specified
    input_dir: str = MISSING
    # Input files relative to input_dir which are used for majority voting
    # Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_dir/output-rs*.jsonl"
    input_files: Any = MISSING

    # Output directory is optional depending on whether the task is to fill the majority answer
    # or to just extract the best answer
    output_dir: str | None = None

    # The script can be run in two modes:
    # 1. fill: use the best answer as the expected_answer to fill input_files
    # 2. extract: identify the best answer from input_files
    mode: str = MISSING

    # where to put the majority answer. By default replacing the expected_answer (assuming it's unknown)
    # but change to predicted_answer, to follow up with a judge evaluation
    fill_key: str = "expected_answer"

    # if True, will not change the fill_key if it's already filled with not None
    ignore_if_not_none: bool = False

    # if True, will use string match to fill is_correct key
    fill_is_correct: bool = True

    # if True, will use the RM score for weighted majority voting
    use_rm_score: bool = False

    # if provided, will fail if can't find this many files. Useful in scheduled
    # pipelines to ensure this step doesn't run if some of the expected files are missing
    require_num_files: int | None = None

    def __post_init__(self):
        """Building data_file from dataset/split if not provided directly."""
        if isinstance(self.input_files, str):
            self.input_files = self.input_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_fill_majority_answer_config", node=FillMajorityAnswerConfig)


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


@hydra.main(version_base=None, config_name="base_fill_majority_answer_config")
def fill_majority_answer(cfg: FillMajorityAnswerConfig):
    cfg = FillMajorityAnswerConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    # Unroll all the files and open them
    input_files = unroll_files(cfg.input_files, parent_dir=cfg.input_dir)
    input_file_handles = [open(file, "rt", encoding="utf-8") for file in input_files]
    if cfg.require_num_files is not None:
        if len(input_file_handles) != cfg.require_num_files:
            raise ValueError(f"Expected {cfg.require_num_files} files, found {len(input_file_handles)}")

    if cfg.mode == "fill":
        if cfg.output_dir is None:
            raise ValueError("output_dir is required when mode is fill")
        # Create output files and their handles with the same relative paths as the input files
        output_files = [
            map_to_output_path(file, cfg.input_dir, cfg.output_dir)
            for file in unroll_files(cfg.input_files, parent_dir=cfg.input_dir)
        ]
        output_file_handles = [open(file, "wt", encoding="utf-8") for file in output_files]
    elif cfg.mode == "extract":
        if cfg.use_rm_score:
            file_suffix = "-best-rm.jsonl"
        else:
            file_suffix = "-best.jsonl"
        output_file_handles = [open(Path(cfg.output_dir) / f"output{file_suffix}", "wt", encoding="utf-8")]
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    new_answers = []
    all_predictions = []
    for idx, predictions in enumerate(tqdm(zip_longest(*input_file_handles))):
        data = read_predictions(predictions, idx, input_file_handles)
        for elem in data:
            if 'predicted_answer' not in elem:
                elem['predicted_answer'] = extract_answer(elem['generation'])
        all_predictions.append(data)

        if not cfg.use_rm_score:
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers = [elem['predicted_answer'] for elem in data if elem['predicted_answer'] is not None]
            new_answers.append(("no_valid_answer_found", (0, len(input_file_handles))))
            if len(valid_answers) == 0:
                continue
            majority_answer, num_votes = Counter(valid_answers).most_common(1)[0]
            new_answers[-1] = (majority_answer, (num_votes, len(input_file_handles)))
        else:
            valid_answers_and_scores = [
                (elem['predicted_answer'], elem['reward_model_score'])
                for elem in data
                if elem['predicted_answer'] is not None
            ]
            new_answers.append(("no_valid_answer_found", 0))
            if len(valid_answers_and_scores) == 0:
                continue

            # Calculate the total score for each answer
            # TODO: This dictionary is just using surface form matching. Need to adapt for semantic matching.
            answer_scores = defaultdict(float)
            for answer, score in valid_answers_and_scores:
                answer_scores[answer] += score

            # Answer is the top-scoring weighted reward model score
            rm_answer, rm_score = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)[0]
            new_answers[-1] = (rm_answer, rm_score)

    for file_handle in input_file_handles:
        file_handle.close()

    if cfg.mode == "fill":
        total_solutions_changed = 0
        total_problems_changed = 0

        for idx, predictions in enumerate(all_predictions):
            changed = False
            for fidx, handle in enumerate(output_file_handles):
                if cfg.ignore_if_not_none and predictions[fidx][cfg.fill_key]:
                    handle.write(json.dumps(predictions[fidx]) + "\n")
                    continue

                if predictions[fidx].get(cfg.fill_key) != new_answers[idx][0]:
                    total_solutions_changed += 1
                    changed = True

                predictions[fidx][cfg.fill_key] = new_answers[idx][0]
                if not cfg.use_rm_score:
                    predictions[fidx]["majority_votes"], predictions[fidx]["total_votes"] = new_answers[idx][1]
                else:
                    predictions[fidx]["answer_rm_score"] = new_answers[idx][1]
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

        # Close all files before moving
        for handle in output_file_handles:
            handle.close()

    elif cfg.mode == "extract":
        best_answer_file_handle = output_file_handles[0]
        with open(input_files[0], "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                data["predicted_answer"] = new_answers[idx][0]
                best_answer_file_handle.write(json.dumps(data) + "\n")

        best_answer_file_handle.close()


HELP_MESSAGE = get_help_message(FillMajorityAnswerConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        fill_majority_answer()
