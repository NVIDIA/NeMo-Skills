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

from typing import Dict


def add_rounding_instruction(data: Dict) -> Dict:
    try:
        float(data['expected_answer'])
        number_of_values = 0
        if '.' in str(data['expected_answer']):
            number_of_values = len(str(data['expected_answer']).split('.')[1])
        if number_of_values == 0:
            data['problem'] += ' Express the answer as an integer.'
        elif number_of_values == 1:
            data['problem'] += ' Round the answer to one decimal place.'
        else:
            data['problem'] += f' Round the answer to {number_of_values} decimal places.'
    except ValueError:
        pass
    return data
