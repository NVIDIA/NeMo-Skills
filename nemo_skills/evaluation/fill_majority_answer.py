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
import sys
from collections import Counter
from itertools import zip_longest
from typing import Any

import hydra
from omegaconf import MISSING
from tqdm import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics import read_predictions
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)

# TODO: rename this script as it's not also handling RM rescoring


@nested_dataclass(kw_only=True)
class FillMajorityAnswerConfig:
    """Top-level parameters for the script"""

    # list of files to use for majority voting.
    # Can specify multiple patterns separated by space
    # e.g. "path/to/file1.jsonl path/to/file2.jsonl" or with regex
    # "test_dir/output-rs*.jsonl"
    input_files: Any = MISSING

    # where to put the majority answer. By default replacing the expected_answer (assuming it's unknown)
    # but change to predicted_answer, to follow up with a judge evaluation
    fill_key: str = "expected_answer"

    # if True, will not change the fill_key if it's already filled with not None
    ignore_if_not_none: bool = False

    # if True, will use string match to fill is_correct key
    fill_is_correct: bool = True

    # if True, will use the highest RM score instead of majority voting
    use_highest_rm_score: bool = False

    def __post_init__(self):
        """Building data_file from dataset/split if not provided directly."""
        if isinstance(self.input_files, str):
            self.input_files = self.input_files.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_fill_majority_answer_config", node=FillMajorityAnswerConfig)


@hydra.main(version_base=None, config_name="base_fill_majority_answer_config")
def fill_majority_answer(cfg: FillMajorityAnswerConfig):
    cfg = FillMajorityAnswerConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(cfg.input_files)]

    new_answers = []
    all_predictions = []
    for idx, predictions in enumerate(tqdm(zip_longest(*file_handles))):
        data = read_predictions(predictions)
        for elem in data:
            if 'predicted_answer' not in elem:
                elem['predicted_answer'] = extract_answer(elem['generation'])
        all_predictions.append(data)

        if not cfg.use_highest_rm_score:
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers = [elem['predicted_answer'] for elem in data if elem['predicted_answer'] is not None]
            new_answers.append((None, (0, len(file_handles))))
            if len(valid_answers) == 0:
                continue
            majority_answer, num_votes = Counter(valid_answers).most_common(1)[0]
            new_answers[-1] = (majority_answer, (num_votes, len(file_handles)))
        else:
            valid_answers_and_scores = [
                (elem['predicted_answer'], elem['reward_model_score'])
                for elem in data
                if elem['predicted_answer'] is not None
            ]
            new_answers.append((None, 0))
            if len(valid_answers_and_scores) == 0:
                continue

            # Answer is the top-scoring reward
            rm_answer, rm_score = sorted(valid_answers_and_scores, key=lambda x: x[1], reverse=True)[0]
            new_answers[-1] = (rm_answer, rm_score)

    for file_handle in file_handles:
        file_handle.close()

    # writing the majority answers back to the files
    file_handles = [open(file, "wt", encoding="utf-8") for file in unroll_files(cfg.input_files)]
    for idx, predictions in enumerate(all_predictions):
        for lidx, handle in enumerate(file_handles):
            if cfg.ignore_if_not_none and predictions[lidx][cfg.fill_key] is not None:
                handle.write(json.dumps(predictions[lidx]) + "\n")
                continue
            predictions[lidx][cfg.fill_key] = new_answers[idx][0]
            if not cfg.use_highest_rm_score:
                predictions[lidx]["majority_votes"], predictions[lidx]["total_votes"] = new_answers[idx][1]
            else:
                predictions[lidx]["answer_rm_score"] = new_answers[idx][1]
            # this is just a string match check, so for full correctness need to rerun the evaluator
            if cfg.fill_is_correct:
                predictions[lidx]["is_correct"] = (
                    predictions[lidx]["predicted_answer"] == predictions[lidx]["expected_answer"]
                )
            else:
                predictions[lidx].pop("is_correct")
            handle.write(json.dumps(predictions[lidx]) + "\n")

    for file_handle in file_handles:
        file_handle.close()


HELP_MESSAGE = get_help_message(FillMajorityAnswerConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        fill_majority_answer()
