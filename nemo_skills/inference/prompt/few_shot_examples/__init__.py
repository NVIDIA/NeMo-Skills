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
from nemo_skills.code_execution import CODE_OUTPUT_SEPARATORS, CODE_SEPARATORS
from nemo_skills.inference.prompt.few_shot_examples.examples_gsm8k import examples_map as examples_gsm8k
from nemo_skills.inference.prompt.few_shot_examples.examples_math import examples_map as examples_math
from nemo_skills.inference.prompt.few_shot_examples.examples_tabmwp import examples_map as examples_tabmwp

examples_map = examples_gsm8k.copy()
examples_map.update(examples_math)
examples_map.update(examples_tabmwp)
assert len(examples_map) == len(examples_gsm8k) + len(examples_math) + len(
    examples_tabmwp
), "Duplicate keys in examples!"


# post-processing to replace code separators with actual tokens
for examples in examples_map.values():
    for example in examples:
        # not using .format to not complicate other {X} parts of the solution
        if 'generation' in example:
            example["generation"] = example["generation"].replace("{start_code}", CODE_SEPARATORS[0])
            example["generation"] = example["generation"].replace("{end_code}", CODE_SEPARATORS[1])
            example["generation"] = example["generation"].replace("{start_code_output}", CODE_OUTPUT_SEPARATORS[0])
            example["generation"] = example["generation"].replace("{end_code_output}", CODE_OUTPUT_SEPARATORS[1])
