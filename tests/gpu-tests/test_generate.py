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

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import ComputeMetrics

# TODO: retrieval test


@pytest.mark.gpu
def test_vllm_generate_greedy():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy/generation "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/test.jsonl "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is no is_correct key
    with open(f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy/generation/output.jsonl") as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    for line in lines:
        data = json.loads(line)
        assert 'is_correct' not in data
        assert 'generation' in data
    assert os.path.exists(f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy/generation/output.jsonl.done")


@pytest.mark.gpu
def test_vllm_generate_greedy_chunked():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy-chunked/generation "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_chunks 2 "
        f"    ++input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/test.jsonl "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is no is_correct key
    with open(f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy-chunked/generation/output.jsonl") as fin:
        lines = fin.readlines()
    assert len(lines) == 20  # because max_samples is the number of samples per chunk
    for line in lines:
        data = json.loads(line)
        assert 'is_correct' not in data
        assert 'generation' in data
    assert os.path.exists(f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy/generation/output.jsonl.done")


@pytest.mark.gpu
def test_vllm_generate_seeds():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    if model_type != 'llama':
        pytest.skip("Only running this test for llama models")

    num_seeds = 3
    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/{model_type}/vllm-generate-seeds/generation "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_random_seeds {num_seeds} "
        f"    --eval_args='++eval_type=math' "
        f"    ++dataset=gsm8k "
        f"    ++split=test "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that all 3 files are created
    for seed in range(num_seeds):
        with open(f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-seeds/generation/output-rs{seed}.jsonl") as fin:
            lines = fin.readlines()
        assert len(lines) == 10
        for line in lines:
            data = json.loads(line)
            assert 'is_correct' in data
            assert 'generation' in data
        assert os.path.exists(
            f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-seeds/generation/output-rs{seed}.jsonl.done"
        )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark='gsm8k').compute_metrics(
        [f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-seeds/generation/output-rs*.jsonl"],
    )["all"]["majority@3"]
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics['symbolic_correct'] >= 50
    assert metrics['num_entries'] == 10
