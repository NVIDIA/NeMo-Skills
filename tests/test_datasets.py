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
    subprocess.run(
        f'python {Path(__file__).absolute().parents[1] / "datasets" / "prepare.py"}', shell=True, check=True
    )

    # checking that all expected files are created
    expected_files = [
        'algebra222/test.jsonl',
        'asdiv/test.jsonl',
        'gsm-hard/test.jsonl',
        'gsm-hard/test_rounded.jsonl',
        'mawps/test.jsonl',
        'svamp/test.jsonl',
        'tabmwp/train.jsonl',
        'tabmwp/validation.jsonl',
        'tabmwp/test.jsonl',
        'gsm8k/train.jsonl',
        'gsm8k/train_full.jsonl',
        'gsm8k/validation.jsonl',
        'gsm8k/validation-sft.jsonl',
        'gsm8k/validation-sft-chat.jsonl',
        'gsm8k/test.jsonl',
        'gsm-plus/test.jsonl',
        'gsm-plus/test_rounded.jsonl',
        'gsm-ic-2step/test.jsonl',
        'gsm-ic-mstep/test.jsonl',
        'functional/test.jsonl',
        'math/train.jsonl',
        'math/train_full.jsonl',
        'math/validation.jsonl',
        'math/validation-sft.jsonl',
        'math/validation-sft-chat.jsonl',
        'math/test.jsonl',
        'gsm8k-masked/train.jsonl',
        'gsm8k-masked/train_full.jsonl',
        'gsm8k-masked/validation.jsonl',
        'gsm8k-masked/validation-sft.jsonl',
        'gsm8k-masked/validation-sft-chat.jsonl',
        'math-masked/train.jsonl',
        'math-masked/train_full.jsonl',
        'math-masked/validation.jsonl',
        'math-masked/validation-sft.jsonl',
        'math-masked/validation-sft-chat.jsonl',
        'human-eval/test.jsonl',
        'mbpp/test.jsonl',
        'mmlu/test.jsonl',
        'mmlu/dev.jsonl',
        'mmlu/val.jsonl',
        'ifeval/test.jsonl',
        'math-odyssey/test.jsonl',
        'aime-2024/test.jsonl',
    ]
    for file in expected_files:
        assert (Path(__file__).absolute().parents[1] / "datasets" / file).exists()
