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
import sys
from pathlib import Path

import hydra
from omegaconf import MISSING

sys.path.append(str(Path(__file__).absolute().parents[2]))

from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass
class PrepareMaskedDataConfig:
    """Top-level parameters for the script"""

    dataset: str = MISSING
    masked_soln_jsonl_files: str = MISSING
    split_name: str = "train_full"

    def __post_init__(self):
        if self.split_name not in ["train", "train_full"]:
            raise ValueError("`split_name` must be one of the following: `train`, `train_full`")
        if isinstance(self.masked_soln_jsonl_files, str):
            self.masked_soln_jsonl_files = self.masked_soln_jsonl_files.split(" ")


def count_digits(input_string):
    digit_count = sum(char.isdigit() for char in input_string)
    return digit_count


def choose_masked_soln_candidate(ref_soln, cand_solns, len_margin=None):
    ref_soln_len = len(ref_soln)
    filt_cand_solns = [cand_soln for cand_soln in cand_solns if cand_soln != ""]

    if len_margin:
        filt_cand_solns = [
            cand_soln
            for cand_soln in cand_solns
            if (
                (len(cand_soln) >= (1 - len_margin) * ref_soln_len)
                and (len(cand_soln) <= (1 + len_margin) * ref_soln_len)
            )
        ]

    if filt_cand_solns:
        edit_distance_list = [(count_digits(cand_soln), cand_soln) for cand_soln in filt_cand_solns]
        edit_distance_list.sort(key=lambda x: x[0])
        return edit_distance_list[0][1]

    return ""


def load_masked_solns(masked_soln_file):
    """Load masked solutions."""
    masked_files = unroll_files(masked_soln_file)

    def load_solns(masked_file):
        """Load masked solutions."""
        masked_solns = []
        masked_corr_solns = []
        with open(masked_file) as masked_f:
            for masked_soln_line in masked_f:
                masked_soln_obj = json.loads(masked_soln_line.strip())
                masked_soln = masked_soln_obj["generated_solution"]

                if masked_soln_obj.get("is_correct", False):
                    # Means that the correct answer could be extracted from the answer string
                    masked_solns.append("")
                    masked_corr_solns.append(masked_soln)
                else:
                    masked_solns.append(masked_soln)
                    masked_corr_solns.append("")

        return masked_solns, masked_corr_solns

    soln_list = []
    soln_list_corr = []

    for masked_file in masked_files:
        masked_solns, masked_corr_solns = load_solns(masked_file)
        soln_list.append(masked_solns)
        soln_list_corr.append(masked_corr_solns)

    comb_soln_list, comb_soln_list_corr = list(zip(*soln_list)), list(zip(*soln_list_corr))
    return comb_soln_list, comb_soln_list_corr


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_prepare_masked_data_config", node=PrepareMaskedDataConfig)


@hydra.main(version_base=None, config_name="base_prepare_masked_data_config")
def prepare_masked_data(cfg: PrepareMaskedDataConfig):
    cfg = PrepareMaskedDataConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    original_data_folder = Path(__file__).parents[2] / "datasets" / cfg.dataset
    masked_data_folder = Path(__file__).parents[2] / "datasets" / (cfg.dataset + "_masked")

    # Input file
    train_file = original_data_folder / (cfg.split_name + ".jsonl")
    # Output file
    output_file = masked_data_folder / (cfg.split_name + ".jsonl")

    masked_solns, masked_corr_solns = load_masked_solns(cfg.masked_soln_jsonl_files)

    problematic_instances = 0
    answer_not_masked_instance = 0
    os.makedirs(masked_data_folder, exist_ok=True)
    with open(train_file, mode="r") as input_f, open(output_file, mode="w") as output_f:
        for idx, train_line in enumerate(input_f):
            train_instance = json.loads(train_line.strip())
            ref_soln = train_instance["reference_solution"].strip()

            cand_soln = choose_masked_soln_candidate(ref_soln, masked_solns[idx], len_margin=0.5)
            if cand_soln == "":
                # Best masked soln is empty string
                answer_not_masked_instance += 1
                # Pick amongst the correct masked solns
                cand_soln = choose_masked_soln_candidate(ref_soln, masked_corr_solns[idx], len_margin=None)

            # Replace reference solution with masked solution
            train_instance["reference_solution"] = ref_soln
            train_instance["reference_masked_solution"] = cand_soln

            if cand_soln == "":
                # Masked solution length is too different from the reference solution
                cand_soln = ref_soln
                problematic_instances += 1

            train_instance["reference_masked_solution"] = cand_soln
            output_f.write(json.dumps(train_instance) + '\n')

        LOG.info("Answer not masked soln: %d", answer_not_masked_instance)
        LOG.info("Length mismatch in soln: %d", problematic_instances)


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        help_msg = get_help_message(PrepareMaskedDataConfig)
        print(help_msg)
    else:
        setup_logging()
        prepare_masked_data()
