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

import subprocess
from pathlib import Path


def test_data_scripts():
    # top-level
    # test is not supported
    cmd = """python nemo_skills/inference/generate_solutions.py \
        output_file=./test-results/gsm8k/output-greedy.jsonl \
        +prompt=code_sfted \
        ++prompt.few_shot_examples.examples_type=null \
        ++prompt.context_type=empty \
        ++dataset=gsm8k \
        ++split_name=test \
        ++server.server_type=nemo \
        ++server.host=1 \
        ++test=1"""
    subprocess.run(
        f'bash {Path(__file__).absolute().parents[1] / "datasets" / "prepare_all.sh"}', shell=True, check=True
    )

    # checking that all expected files are created
    expected_files = [
        'algebra222/test.jsonl',
        'asdiv/test.jsonl',
        'gsm-hard/test.jsonl',
        'mawps/test.jsonl',
        'svamp/test.jsonl',
        'tabmwp/test.jsonl',
        'gsm8k/train.jsonl',
        'gsm8k/train_full.jsonl',
        'gsm8k/validation.jsonl',
        'gsm8k/validation-sft.jsonl',
        'gsm8k/test.jsonl',
        'math/train.jsonl',
        'math/train_full.jsonl',
        'math/validation.jsonl',
        'math/validation-sft.jsonl',
        'math/test.jsonl',
        'gsm8k-masked/train.jsonl',
        'gsm8k-masked/train_full.jsonl',
        'gsm8k-masked/validation.jsonl',
        'gsm8k-masked/validation-sft.jsonl',
        'math-masked/train.jsonl',
        'math-masked/train_full.jsonl',
        'math-masked/validation.jsonl',
        'math-masked/validation-sft.jsonl',
    ]
    for file in expected_files:
        assert (Path(__file__).absolute().parents[1] / "datasets" / file).exists()
