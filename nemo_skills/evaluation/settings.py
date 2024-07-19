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

# a collection of settings required to correctly running evaluations for different benchmarks
# in addition to what's in this file, there is also some prompt engineering that needs
# to happen in eval_map.py inside prompt folder for specific models

from nemo_skills.evaluation.graders import arena_grader, code_grader, if_grader, math_grader
from nemo_skills.evaluation.metrics import ArenaEval, CodeEval, IFEval, MathEval

MATH_BENCHMARKS = [
    'algebra222',
    'asdiv',
    'functional',
    'gsm-hard',
    'gsm-ic-2step',
    'gsm-ic-mstep',
    'gsm-plus',
    'gsm8k',
    'math',
    'mawps',
    'svamp',
    'tabmwp',
    'math-odyssey',
    'aime-2024',
]
CODE_BENCHMARKS = ['human-eval', 'mbpp']


# ------------------------------- metrics settings -----------------------------
EVALUATOR_MAP = {
    "ifeval": IFEval,
    "arena-hard": ArenaEval,
    "mmlu": MathEval,  # TODO: update this
}

for benchmark in MATH_BENCHMARKS:
    EVALUATOR_MAP[benchmark] = MathEval

for benchmark in CODE_BENCHMARKS:
    EVALUATOR_MAP[benchmark] = CodeEval
# ------------------------------------------------------------------------------

# -------------------------------- eval settings -------------------------------
EXTRA_EVAL_ARGS = {
    # some benchmarks require specific extra arguments, which are defined here
    'human-eval': '++eval_type=code ++eval_config.dataset=humaneval',
    'mbpp': '++eval_type=code ++eval_config.dataset=mbpp',
    'ifeval': '++eval_type=ifeval',
    'arena-hard': '++eval_type=arena',
}

# TODO: better name?
GRADING_MAP = {
    "math": math_grader,  # that's default. TODO: should we do this per-benchmark?
    "code": code_grader,
    "ifeval": if_grader,
    "arena": arena_grader,
}
# ------------------------------------------------------------------------------

# --------------------------------- gen settings -------------------------------
EXTRA_GENERATION_ARGS = {
    # some benchmarks require specific extra arguments, which are defined here
    'ifeval': '++generation_key=response',
}
# ------------------------------------------------------------------------------
