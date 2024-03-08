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

import sys
from pathlib import Path

import yaml

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config


def prepare_for_sft(data, prompt_type, dataset):
    # reading prompt format from the yaml file
    prompt_config = get_prompt_config(prompt_type)
    prompt_config.context_type = "empty"
    prompt_config.few_shot_examples.num_few_shots = 0

    prepared_data = []
    for original_elem in data:
        elem = {}
        elem["input"] = str(Prompt(prompt_config, original_elem))
        # note that the loss will not be meaningful,
        # since our solution format is different, but we need to populate that field
        elem["output"] = original_elem['reference_solution']
        elem["expected_answer"] = original_elem['expected_answer']
        elem["dataset"] = dataset
        prepared_data.append(elem)
    return prepared_data
