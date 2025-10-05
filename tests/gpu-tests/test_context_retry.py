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
import tempfile
import time
from pathlib import Path

import pytest

from tests.conftest import docker_rm, docker_run

ACCURACY_THRESHOLD = 25
MAX_EVAL_SAMPLES = 4


def _create_large_input_file(input_file: str, num_samples: int):
    """Create a fake input jsonl file with long prompts"""
    # TODO: Currently this is just a single turn message. Need to add tests for multi-turn messages.
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            for _ in range(num_samples):
                # Create large prompt that will likely exceed context limits
                large_prompt = {"question": "a" * 500_000 + "b" * 500_000}
                temp_file.write(json.dumps(large_prompt).encode())
                temp_file.write(b"\n")
            temp_file.flush()

            # Copy to Docker container
            docker_run(f"cp {temp_file.name} {input_file}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file.name)


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["vllm"])  # add "sglang" / "trtllm" when supported similarly
def test_context_retry(server_type):
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    cfg_dir = Path(__file__).absolute().parent
    base_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-test_context_retry"
    eval_dir = f"{base_dir}/eval"
    # eval_dir = f"{base_dir}/eval"
    # input_file = f"{base_dir}/input.jsonl"

    # with open(input_file, "w") as f:
    #     f.write(json.dumps({"question": "What is 2+2?"}) + "\n")

    docker_rm([base_dir])

    # 1) Start server (fixed port via --get_random_port=False for deterministic client targeting)
    server_cmd = (
        f"ns start_server "
        f"  --cluster test-local --config_dir {cfg_dir} "
        f"  --model {model_path} "
        f"  --server_type {server_type} "
        f"  --server_gpus 1 "
        f"  --server_nodes 1 "
    )
    if server_type == "trtllm":
        server_cmd += " --server_args '--backend pytorch'"

    server_proc = subprocess.Popen(server_cmd, shell=True)

    # Wait for server to be ready
    time.sleep(60)

    try:
        # (1) With no soft fail, the generation should fail
        docker_rm([eval_dir])
        base_eval_cmd = (
            f"ns eval "
            f"  --cluster test-local --config_dir {cfg_dir} "
            f"  --server_address http://0.0.0.0:5000/v1 "
            f"  --server_type {server_type} "
            f"  --output_dir {eval_dir} "
            f"  --benchmarks gsm8k "
            f"  --model {model_path} "
            f"  ++max_samples={MAX_EVAL_SAMPLES} "
        )
        subprocess.run(base_eval_cmd, shell=True, check=True)
        # Check that the eval output is not created
        assert not os.path.exists(f"{eval_dir}/output.jsonl"), "Eval output should not be done"

        # (2) With soft fail, the eval should succeed
        docker_rm([eval_dir])

        eval_cmd = base_eval_cmd + " ++server.enable_soft_fail=True"
        subprocess.run(eval_cmd, shell=True, check=True)

        # Check that the eval output is created
        metrics_file = f"{eval_dir}/eval-results/gsm8k/metrics.json"
        assert os.path.exists(metrics_file), "Metrics file should be created with soft fail"
        # Check that the number of samples is 4 but with 0 performance
        with open(metrics_file, "r") as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]
        assert metrics["num_entries"] == 4
        assert metrics["symbolic_correct"] == 0

        # 3) With soft fail and reduce_generation strategy, the eval should succeed
        docker_rm([eval_dir])

        eval_cmd = base_eval_cmd + (
            " ++server.enable_soft_fail=True ++server.context_limit_retry_strategy=reduce_generation"
        )
        subprocess.run(eval_cmd, shell=True, check=True)

        # Check that the eval output is created
        metrics_file = f"{eval_dir}/eval-results/gsm8k/metrics.json"
        assert os.path.exists(metrics_file), "Metrics file should be created with soft fail"
        # Check that the number of samples is 4 but with 0 performance
        with open(metrics_file, "r") as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]
        assert metrics["num_entries"] == 4
        assert metrics["symbolic_correct"] == ACCURACY_THRESHOLD

    finally:
        # best-effort teardown of the CLI process
        try:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        except Exception:
            pass
