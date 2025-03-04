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

examples_map = examples_gsm8k.copy()
examples_map.update(examples_math)
examples_map.update(examples_lean4)
examples_map.update(examples_mmlu_pro)
assert len(examples_map) == len(examples_gsm8k) + len(examples_math) + len(
    examples_lean4
) + len(examples_mmlu_pro), "Duplicate keys in examples!"
