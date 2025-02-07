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

import contextlib
import os
import time
import requests
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt


def wait_for_server(url, timeout=300, interval=1):
    """
    Waits for a server to be up and running.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.put(url)
            if response.status_code != 403:  # Check if server responds
                return True
        except requests.RequestException:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not respond within timeout period")
        time.sleep(interval)

def start_server(model_path: str, server_type: str) -> None:
    """
    Starts a server for a given model and server type.
    We then wait a few seconds for the server to be up.
    """
    cmd = (
        f"ns start_server "
        f"--cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"--model {model_path} "
        f"--server_type {server_type} "
        f"--server_gpus 1 "
        f"--server_nodes 1"
    )
    proc = subprocess.Popen(cmd, shell=True)

    return proc

@contextlib.contextmanager
def managed_server(model_path, server_type):
    proc = start_server(model_path, server_type)
    try:
        wait_for_server("http://127.0.0.1:5000")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except TimeoutError:
            proc.kill()
            proc.wait()

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
    prompt = get_prompt('generic/default', prompt_template)
    prompts = [prompt.fill({'question': "What is the answer to life, the universe and everything?"})]
    for server_type, model_path in model_info:
        if not model_path:
            continue

        with managed_server(model_path, server_type):
            llm = get_model(server_type=server_type, host="localhost")
            outputs = llm.generate(prompts=prompts, tokens_to_generate=15, top_logprobs=1, temperature=0.5)
            output = outputs[0]
            _test_individual_generations(output, server_type)

            logprobs = output["logprobs"]
            tokens = output["tokens"]
            outputs_map[server_type] = list(zip(tokens, logprobs))

    if "vllm" not in outputs_map or "nemo" not in outputs_map:
        pytest.skip("Not enough models available to compare top_logprobs consistency")

    # trtllm for some reason produces quite different outputs so we don't compare to it
    server_type = "vllm"
    other_server_type = "nemo"
    tolerance = 0.5
    for (token, logprob), (other_token, other_logprob) in zip(outputs_map[server_type], outputs_map[other_server_type]):
        assert token == other_token, f"Tokens for {server_type} and {other_server_type} do not match: '{token}' vs '{other_token}'"
        assert abs(logprob - other_logprob) < tolerance, f"Logprobs for {server_type} and {other_server_type} do not match for token '{token}': {logprob} vs {other_logprob}"
