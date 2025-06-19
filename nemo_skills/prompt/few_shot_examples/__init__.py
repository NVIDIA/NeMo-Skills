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
from nemo_skills.prompt.few_shot_examples.gsm8k import examples_map as examples_gsm8k
from nemo_skills.prompt.few_shot_examples.math import examples_map as examples_math
from nemo_skills.prompt.few_shot_examples.lean4 import examples_map as examples_lean4
from nemo_skills.prompt.few_shot_examples.mmlu_pro import examples_map as examples_mmlu_pro
from nemo_skills.prompt.few_shot_examples.mmlu import examples_map as examples_mmlu

all_example_sets = [
    examples_gsm8k,
    examples_math,
    examples_lean4,
    examples_mmlu_pro,
    examples_mmlu,
]

examples_map = {k: v for d in all_example_sets for k, v in d.items()}

# Verify no duplicate keys exist across example sets
expected_total_examples = sum(len(example_set) for example_set in all_example_sets)
assert len(examples_map) == expected_total_examples, f"Duplicate keys in examples!"
