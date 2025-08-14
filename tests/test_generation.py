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

import json
import os

# running most things through subprocess since that's how it's usually used
import subprocess
from pathlib import Path

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics

DATA_TO_TEST = []
template_folder = Path(__file__).parents[1] / 'nemo_skills' / 'prompt' / 'template'

for dataset, split in [('gsm8k', 'train'), ('gsm8k', 'test'), ('math-500', 'test')]:
    DATA_TO_TEST.append((dataset, split))


def test_eval_gsm8k_api(tmp_path):
    if not os.getenv('NVIDIA_API_KEY'):
        pytest.skip("Define NVIDIA_API_KEY to run this test")

    cmd = (
        f"ns eval "
        f"    --server_type=azureopenai "
        f"    --model=gpt-4.1-20250414 "
        f"    --server_address=https://llm-proxy.perflab.nvidia.com "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark='gsm8k').compute_metrics(
        [f"{tmp_path}/eval-results/gsm8k/output.jsonl"],
    )[
        "_all_"
    ]["pass@1"]

    assert metrics['symbolic_correct'] >= 80


@pytest.mark.parametrize("format", ["list", "dict"])
def test_generate_openai_format(tmp_path, format):
    if not os.getenv('NVIDIA_API_KEY'):
        pytest.skip("Define NVIDIA_API_KEY to run this test")

    cmd = (
        f"ns generate "
        f"    --server_type=azureopenai "
        f"    --model=gpt-4.1-20250414 "
        f"    --server_address=https://llm-proxy.perflab.nvidia.com "
        f"    --input_file=/nemo_run/code/tests/data/openai-input-{format}.test "
        f"    --output_dir={tmp_path} "
        f"    ++prompt_format=openai "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that output exists and has the expected format
    with open(f"{tmp_path}/output.jsonl") as fin:
        data = [json.loads(line) for line in fin.readlines()]
    assert len(data) == 2
    assert len(data[0]['generation']) > 0
    assert len(data[1]['generation']) > 0
