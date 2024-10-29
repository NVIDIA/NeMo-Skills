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
from nemo_skills.evaluation.metrics import compute_metrics


@pytest.mark.gpu
def test_trtllm_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir /tmp/nemo-skills-tests/trtllm-eval "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=20 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    metrics_calculator = importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS()
    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/trtllm-eval/eval-results/gsm8k/output-greedy.jsonl"], metrics_calculator
    )
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics['symbolic_correct'] >= 50
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_trtllm_code_execution_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir /tmp/nemo-skills-tests/trtllm-eval "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-base "  # we are using the base prompt to make the model follow few-shots
        f"    ++examples_type=gsm8k_text_with_code "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=20 "
        f"    ++code_execution=True "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    metrics_calculator = importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS()
    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/trtllm-eval/eval-results/gsm8k/output-greedy.jsonl"], metrics_calculator
    )
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics['symbolic_correct'] >= 50
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_vllm_eval():
    # this test expects llama3-instruct to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/ifeval will be cut to 400 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir /tmp/nemo-skills-tests/vllm-eval "
        f"    --benchmarks algebra222:0,human-eval:0,mbpp:0,ifeval:0,mmlu:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_jobs 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=400 "
        f"    ++max_samples=400 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        "ns summarize_results /tmp/nemo-skills-tests/vllm-eval",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/algebra222/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.math').METRICS_CLASS(),
    )

    assert metrics['symbolic_correct'] >= 80
    assert metrics['num_entries'] == 222

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/human-eval/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.human-eval').METRICS_CLASS(),
    )
    assert metrics['passing_base_tests'] >= 50
    assert metrics['passing_plus_tests'] >= 50
    assert metrics['num_entries'] == 164

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/mbpp/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.mbpp').METRICS_CLASS(),
    )
    assert metrics['passing_base_tests'] >= 50
    assert metrics['passing_plus_tests'] >= 50
    assert metrics['num_entries'] == 378

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/ifeval/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.ifeval').METRICS_CLASS(),
    )
    assert metrics['prompt_strict_accuracy'] >= 60
    assert metrics['instruction_strict_accuracy'] >= 70
    assert metrics['prompt_loose_accuracy'] >= 60
    assert metrics['instruction_loose_accuracy'] >= 70
    assert metrics['num_prompts'] == 400
    assert metrics['num_instructions'] == 601

    metrics = compute_metrics(
        ["/tmp/nemo-skills-tests/vllm-eval/eval-results/mmlu/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.mmlu').METRICS_CLASS(),
    )
    assert metrics['symbolic_correct'] >= 60
    assert metrics['num_entries'] == 400


@pytest.mark.gpu
def test_nemo_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type nemo "
        f"    --output_dir /tmp/nemo-skills-tests/nemo-eval "
        f"    --benchmarks gsm8k:0 "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template=llama3-instruct "
        f"    ++split=test "
        f"    ++batch_size=8 "
        f"    ++max_samples=20 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    metrics_calculator = importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS()
    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/nemo-eval/eval-results/gsm8k/output-greedy.jsonl"], metrics_calculator
    )
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics['symbolic_correct'] >= 50
    assert metrics['num_entries'] == 20
