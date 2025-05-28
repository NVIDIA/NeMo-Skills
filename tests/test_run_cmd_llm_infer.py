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
import sys
import time
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))


def test_run_cmd_llm_infer():
    """
    Starts (if available) TRTLLM, Nemo, and VLLM servers, then sends the same prompt
    with top_logprobs=1. It then compares the logprobs for each token across the models.
    """
    model_info = [
        ("vllm", 'Qwen/Qwen2.5-0.5B-Instruct'),
    ]

    outputs_map = {}
    for server_type, model_path in model_info:
        if not model_path:
            continue

        output_dir = f"/tmp/nemo-skills-tests/qwen2.5/{server_type}-run-cmd"
        command = (
            f"cd /nemo_run/code/tests/tests/scripts/ && "
            f"mkdir -p {output_dir} && "
            f"python run_cmd_llm_infer_check.py > {output_dir}/output.txt"
        )

        cmd = (
            f"ns run_cmd "
            f"--cluster test-local --config_dir {Path(__file__).absolute().parent / 'gpu-tests'} "
            f"--model {model_path} "
            f"--server_type {server_type} "
            f"--server_gpus 0 "
            f"--server_nodes 1 "
            f"--command '{command}'"
        )
        subprocess.run(cmd, shell=True, check=True)
        time.sleep(120)  # Wait for the server to finish generating
        jsonl_file = Path(output_dir) / "output.txt"

        with open(jsonl_file, "r") as f:
            outputs = f.read()

        assert len(outputs) > 0  # just check that output text is not zero.
