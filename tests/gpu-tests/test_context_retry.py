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
import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_context_retry_reduce_generation():
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
        f"    ++max_samples=10 "
        f"    ++inference.tokens_to_generate=1_000_000 "
        f"    ++server.enable_soft_fail=True "
        f"    ++server.context_limit_retry_strategy=reduce_generation "
    )

    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]

    # rough check, since exact accuracy varies depending on gpu type
    if model_type == "llama":
        assert metrics["symbolic_correct"] >= 30
    else:  # qwen
        assert metrics["symbolic_correct"] >= 30
    assert metrics["num_entries"] == 10
