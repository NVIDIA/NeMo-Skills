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

# need to contain an EVAL_MAP dictionary mapping model version name to prompts for different benchmarks
# can have a default key that will be used if benchmark name is not in dict

EVAL_MAP = {
    'base': {
        'default': 'openmathinstruct/base',
    },
    'instruct': {  # nemotron-instruct
        'default': 'nemotron/math',  # Default to math prompt
        # Math benchmarks
        'gsm8k': 'nemotron/math',
        'math': 'nemotron/math',
        # Coding
        'human-eval': 'nemotron/codegen',
        'mbpp': 'nemotron/codegen',
        # Others
        'mmlu': 'nemotron/mmlu',
        'ifeval': 'nemotron/sft',
        'arena-hard': 'nemotron/sft',
    },
}
