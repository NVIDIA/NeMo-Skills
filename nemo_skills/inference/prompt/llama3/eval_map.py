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
    'base': {  # llama3-base
        'default': 'llama3/base',
        # currently code/mmlu are not supported, need to add proper prompts
    },
    'instruct': {  # llama3-instruct
        'default': 'llama3/instruct',
        'gsm8k': 'llama3/gsm8k',
        'math': 'llama3/math',
        # TODO: put proper coding prompts here as well
        'human-eval': 'llama3/codegen',
        'mbpp': 'llama3/codegen',
        'mmlu': 'llama3/mmlu',
        'ifeval': 'llama3/sft',
        'arena-hard': 'llama3/sft',
    },
    'instruct-nemo': {  # llama3-instruct finetuned with nemo_skills
        'default': 'llama3/sft',
        'human-eval': 'llama3/codegen',
        'mbpp': 'llama3/codegen',
        'mmlu': 'llama3/mmlu',
    },
}
