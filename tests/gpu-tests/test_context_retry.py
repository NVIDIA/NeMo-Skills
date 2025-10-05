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
import subprocess
from pathlib import Path

import pytest

from tests.conftest import docker_rm

NUM_TOKENS_TO_GENERATE = 1_000_000
NUM_SAMPLES = 4
ACCURACY_THRESHOLD_PERCENT = 25


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang", "vllm", "trtllm"])
def test_context_retry_no_strategy(server_type):
    """Test that the generation finishes successfully but with error if soft fail is enabled and there is no retry strategy."""
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval-no-strategy"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --benchmarks gsm8k "
        f"    ++max_samples={NUM_SAMPLES} "
        f"    ++inference.tokens_to_generate={NUM_TOKENS_TO_GENERATE} "
        f"    ++server.enable_soft_fail=True "
    )
    cmd += "--server_args '--backend pytorch'" if server_type == "trtllm" else ""
    subprocess.run(cmd, shell=True, check=True)

    metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"
    assert os.path.exists(metrics_file), "Metrics file not found"
    with open(metrics_file) as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    assert metrics["num_entries"] == NUM_SAMPLES
    assert metrics["symbolic_correct"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang", "vllm"])
def test_context_retry_reduce_generation_enabled(server_type):
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval-reduce-generation-enabled"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --benchmarks gsm8k "
        f"    ++max_samples={NUM_SAMPLES} "
        f"    ++inference.tokens_to_generate={NUM_TOKENS_TO_GENERATE} "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_generation "
    )
    subprocess.run(cmd, shell=True, check=True)

    metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"
    assert os.path.exists(metrics_file), "Metrics file not found"
    with open(metrics_file) as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    assert metrics["num_entries"] == NUM_SAMPLES
    assert metrics["symbolic_correct"] >= ACCURACY_THRESHOLD_PERCENT


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["trtllm", "sglang", "vllm"])
def test_context_retry_disabled(server_type):
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval-reduce-generation-disabled"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --benchmarks gsm8k "
        f"    ++max_samples={NUM_SAMPLES} "
        f"    ++inference.tokens_to_generate={NUM_TOKENS_TO_GENERATE} "
        f"    ++server.enable_soft_fail=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"
    if not os.path.exists(metrics_file):
        assert True  # expected: failure means no metrics
    else:
        with open(metrics_file) as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]
        assert metrics["num_entries"] != NUM_SAMPLES, "Expected test to fail but it succeeded"


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["vllm"])
def test_context_retry_reduce_prompt_start(server_type):
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    base_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval-reduce-prompt-start"
    output_dir = f"{base_dir}/generation"
    input_file = f"{base_dir}/input.jsonl"
    docker_rm([base_dir])

    os.makedirs(base_dir, exist_ok=True)
    with open(input_file, "w") as f:
        f.write(json.dumps({"question": "a" * 500_000 + "b" * 500_000}) + "\n")

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --input_file {input_file} "
        f"    ++prompt_config=generic/default "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_prompt_from_start "
        f"    ++inference.tokens_to_generate=2048 "
    )
    subprocess.run(cmd, shell=True, check=True)

    assert os.path.exists(f"{output_dir}/output.jsonl"), "Output file not found"
    assert os.path.exists(f"{output_dir}/output.jsonl.done"), "Done sentinel not found"


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang"])
def test_context_retry_reduce_prompt_end(server_type):
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    base_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval-reduce-prompt-end"
    output_dir = f"{base_dir}/generation"
    input_file = f"{base_dir}/input.jsonl"
    docker_rm([base_dir])

    os.makedirs(base_dir, exist_ok=True)
    with open(input_file, "w") as f:
        f.write(json.dumps({"question": "a" * 500_000 + "b" * 500_000}) + "\n")

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --input_file {input_file} "
        f"    ++prompt_config=generic/default "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_prompt_from_end "
        f"    ++inference.tokens_to_generate=2048 "
    )
    subprocess.run(cmd, shell=True, check=True)

    assert os.path.exists(f"{output_dir}/output.jsonl"), "Output file not found"
    assert os.path.exists(f"{output_dir}/output.jsonl.done"), "Done sentinel not found"
