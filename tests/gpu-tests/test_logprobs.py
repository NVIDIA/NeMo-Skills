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

def _test_individual_generations(output: dict, server_type: str):
    """
    Tests that the output of a model generation has the expected keys, types, and lengths.
    """
    for key in ["generation", "logprobs", "tokens", "num_generated_tokens"]:
        assert key in output, f"{server_type} output is missing '{key}'"
    logprobs = output["logprobs"]
    tokens = output["tokens"]
    assert isinstance(logprobs, list), f"{server_type}: 'logprobs' is not a list"
    assert isinstance(tokens, list), f"{server_type}: 'tokens' is not a list"
    assert len(logprobs) == len(tokens), f"{server_type}: Length of 'logprobs' and 'tokens' do not match"
    assert len(tokens) == output["num_generated_tokens"], f"{server_type}: Length of tokens does not match num_generated_tokens"

@pytest.mark.gpu
def test_cross_model_logprobs_consistency():
    """
    Starts (if available) TRTLLM, Nemo, and VLLM servers, then sends the same prompt
    with top_logprobs=1. It then compares the logprobs for each token across the models.
    """
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    model_info = [
        ("trtllm", os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')),
        ("nemo", os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')),
        ("vllm", os.getenv('NEMO_SKILLS_TEST_HF_MODEL')),
    ]

    outputs_map = {}
    for server_type, model_path in model_info:
        if not model_path:
            continue
        
        output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
        cmd = (
            f"ns eval "
            f"--cluster test-local --config_dir {Path(__file__).absolute().parent} "
            f"--model {model_path} "
            f"--server_type {server_type} "
            f"--output_dir {output_dir} "
            f"--benchmarks gsm8k:1 "
            f"--server_gpus 1 "
            f"--server_nodes 1 "
            f"--skip_greedy "
            f"++prompt_template={prompt_template} "
            f"++split=test "
            f"++batch_size=8 "
            f"++max_samples=20 "
            f"++inference.top_logprobs=1 "
            f"++inference.tokens_to_generate=20 "
            f"++inference.temperature=0.7 "
        )
        subprocess.run(cmd, shell=True, check=True)
        time.sleep(120) # Wait for the server to finish generating
        jsonl_file = Path(output_dir) / "eval-results" / "gsm8k" / "output-rs0.jsonl"

        with open(jsonl_file, "r") as f:
            outputs = [json.loads(line) for line in f.readlines()]
        for output in outputs:
            _test_individual_generations(output, server_type)
        
        output = outputs[0]
        logprobs = output["logprobs"]
        tokens = output["tokens"]
        outputs_map[server_type] = list(zip(tokens, logprobs))

    if "vllm" not in outputs_map or "nemo" not in outputs_map:
        pytest.skip("Not enough models available to compare top_logprobs consistency")

    tolerance = 0.5
    for server_type, _ in model_info:
        for other_server_type, _ in model_info:
            if server_type == other_server_type:
                continue
            assert len(outputs_map[server_type]) == len(outputs_map[other_server_type]), f"Length of outputs do not match between {server_type} and {other_server_type}: {len(outputs_map[server_type])} vs {len(outputs_map[other_server_type])}"
            for (token, logprob), (other_token, other_logprob) in zip(outputs_map[server_type], outputs_map[other_server_type]):
                assert token.replace("Ġ", " ") == other_token.replace("Ġ", " "), f"Tokens for {server_type} and {other_server_type} do not match: '{token}' vs '{other_token}'"
                assert abs(logprob - other_logprob) < tolerance, f"Logprobs for {server_type} and {other_server_type} do not match for token '{token}': {logprob} vs {other_logprob}"
