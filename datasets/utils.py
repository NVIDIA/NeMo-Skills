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
from typing import Dict, List

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config


def prepare_for_sft(data, prompt_type, dataset, chat_format=False):
    # reading prompt format from the yaml file
    prompt_config = get_prompt_config(prompt_type)
    prompt_config.context_type = "empty"
    prompt_config.few_shot_examples.num_few_shots = 0
    prompt = Prompt(config=prompt_config)

    prepared_data = []
    for original_elem in data:
        elem = {}
        # note that the loss will not be meaningful, since we are using reference solution for output
        # and our solution format is different, but we need to populate that field for the code to work
        if chat_format:
            elem['conversations'] = [
                {'value': original_elem['question'], 'from': 'User', 'canonical_form': ''},
                {'value': original_elem["reference_solution"], 'from': 'Assistant', 'canonical_form': ''},
            ]
            elem['system'] = prompt_config.system
            elem['mask'] = 'User'
            elem['type'] = None
        else:
            elem["input"] = prompt.build_string(input_dict=original_elem)
            elem["output"] = original_elem['reference_solution']
        elem["expected_answer"] = original_elem['expected_answer']
        elem["dataset"] = dataset
        prepared_data.append(elem)
    return prepared_data


def add_rounding_instruction(data: Dict) -> Dict:
    try:
        float(data['expected_answer'])
        number_of_values = 0
        if '.' in str(data['expected_answer']):
            number_of_values = len(str(data['expected_answer']).split('.')[1])
        if number_of_values == 0:
            data['question'] += ' Express the answer as an integer.'
        elif number_of_values == 1:
            data['question'] += ' Round the answer to one decimal place.'
        else:
            data['question'] += f' Round the answer to {number_of_values} decimal places.'
    except ValueError:
        pass
    return data
