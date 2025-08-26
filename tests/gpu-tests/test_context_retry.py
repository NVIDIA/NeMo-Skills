# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from nemo_skills.utils import get_logger_name
from tests.conftest import docker_rm, docker_rm_and_mkdir, docker_run


@pytest.mark.gpu
def test_context_retry_reduce_generation_enabled():
    """Test that the generation finishes successfully if soft fail is enabled and the strategy is reduce_generation."""

    NUM_TOKENS_TO_GENERATE = 1_000_000
    NUM_SAMPLES = 4

    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++max_samples={NUM_SAMPLES} "
        f"    ++inference.tokens_to_generate={NUM_TOKENS_TO_GENERATE} "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_generation "
    )

    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]

    # Rough check, at least one correct out of four samples
    accuracy_threshold = 100 // NUM_SAMPLES
    if model_type == "llama":
        assert metrics["symbolic_correct"] >= accuracy_threshold
    else:  # qwen
        assert metrics["symbolic_correct"] >= accuracy_threshold
    assert metrics["num_entries"] == NUM_SAMPLES


@pytest.mark.gpu
def test_context_retry_reduce_generation_disabled():
    """Test that the generation doesn't finish successfully if soft fail is disabled."""

    NUM_TOKENS_TO_GENERATE = 1_000_000
    NUM_SAMPLES = 4

    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++max_samples={NUM_SAMPLES} "
        f"    ++inference.tokens_to_generate={NUM_TOKENS_TO_GENERATE} "
        f"    ++server.enable_soft_fail=False "
        f"    ++server.context_limit_retry_strategy=reduce_generation "
    )

    subprocess.run(cmd, shell=True, check=True)

    metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]

        assert metrics["num_entries"] != NUM_SAMPLES
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_file}")


def _create_fake_big_input_file(input_file, num_samples):
    """Create a fake input jsonl file with long multi-turn prompts."""
    temp_filename = None
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for i in range(num_samples):
            random_prompt = {"question": "aaa " * 1_000_000 + "bbb " * 1_000_000}
            temp_file.write(json.dumps(random_prompt).encode())
            temp_file.write(b"\n")
        temp_file.flush()
        temp_filename = temp_file.name

    docker_command = f"cp {temp_filename} {input_file}"
    docker_run(docker_command)


@pytest.mark.gpu
def test_context_retry_reduce_prompt_start():
    """Test that successful generation is possible if soft fail is enabled and the strategy is reduce_prompt, in this test we remove tokens from the start of the prompt."""

    NUM_SAMPLES = 2

    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-eval-reduce-prompt-start"
    input_file = f"{output_dir}/input.jsonl"
    output_file = f"{output_dir}/output.jsonl"
    docker_rm_and_mkdir(input_file)
    docker_rm_and_mkdir(output_file)

    _create_fake_big_input_file(input_file, num_samples=NUM_SAMPLES)

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir {output_dir} "
        f"    --input_file {input_file} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_config=generic/default "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_prompt_from_start "
    )

    subprocess.run(cmd, shell=True, check=True)

    docker_run(f"ls {output_dir}")
    assert os.path.exists(output_file), "Output file not found"
