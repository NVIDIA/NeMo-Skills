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

# needs to define NEMO_SKILLS_TEST_TRTLLM_MODEL to run these tests
# needs to define NEMO_SKILLS_TEST_NEMO_MODEL to run these tests
# you'd also need 2+ GPUs to run this test
# the metrics are assuming llama3-8b-base as the model and will fail for other models

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import compute_metrics
from nemo_skills.evaluation.settings import EVALUATOR_MAP


@pytest.mark.gpu
def test_trtllm_run_eval():
    model_path = os.getenv('LLAMA3_8B_BASE_TRTLLM')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_TRTLLM') to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type tensorrt_llm \
    --output_dir {output_path} \
    --benchmarks gsm8k:0 \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/base \
    ++prompt.few_shot_examples.examples_type=gsm8k_only_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++split_name=test \
    ++server.code_execution.stop_on_code_error=False \
    ++batch_size=8 \
    ++max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/gsm8k/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generation']
        assert elem['error_message'] != '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/gsm8k/output-greedy.jsonl"], EVALUATOR_MAP['gsm8k']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (40, 5)
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_vllm_run_eval():
    # this test expects llama3-instruct to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/ifeval will be cut to 400 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = os.getenv('LLAMA3_8B_INSTRUCT_HF')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_INSTRUCT_HF to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp') + '/multi-benchmark'

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type vllm \
    --output_dir {output_path} \
    --benchmarks algebra222:0 human-eval:0 mbpp:0 ifeval:0 mmlu:0 \
    --prompt_folder llama3 --model_version instruct \
    --num_gpus 1 \
    --num_nodes 1 \
    --num_jobs 1 \
    ++split_name=test \
    ++batch_size=400 \
    ++max_samples=400 \
"""
    subprocess.run(cmd, shell=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(f"python pipeline/summarize_results.py {output_path}", shell=True, check=True)

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/algebra222/output-greedy.jsonl"], EVALUATOR_MAP['algebra222']())

    assert round(metrics['sympy_correct'], 2) == 66.67
    assert round(metrics['no_answer'], 2) == 1.35
    assert metrics['num_entries'] == 222

    metrics = compute_metrics([f"{output_path}/human-eval/output-greedy.jsonl"], EVALUATOR_MAP['human-eval']())
    assert round(metrics['passing_base_tests'], 2) == 59.76
    assert round(metrics['passing_plus_tests'], 2) == 54.27
    assert metrics['num_entries'] == 164

    metrics = compute_metrics([f"{output_path}/mbpp/output-greedy.jsonl"], EVALUATOR_MAP['mbpp']())
    assert round(metrics['passing_base_tests'], 2) == 69.31
    assert round(metrics['passing_plus_tests'], 2) == 57.41
    assert metrics['num_entries'] == 378

    metrics = compute_metrics([f"{output_path}/ifeval/output-greedy.jsonl"], EVALUATOR_MAP['ifeval']())
    assert abs(metrics['prompt_strict_accuracy'] - 66.50) <= 1.0  # TODO: some randomness in this benchmark
    assert abs(metrics['instruction_strict_accuracy'] - 74.88) <= 1.0
    assert abs(metrics['prompt_loose_accuracy'] - 72.75) <= 1.0
    assert abs(metrics['instruction_loose_accuracy'] - 79.70) <= 1.0
    assert metrics['num_prompts'] == 400
    assert metrics['num_instructions'] == 601

    metrics = compute_metrics([f"{output_path}/mmlu/output-greedy.jsonl"], EVALUATOR_MAP['mmlu']())
    assert round(metrics['sympy_correct'], 2) == 58.75
    assert round(metrics['no_answer'], 2) == 19.75
    assert metrics['num_entries'] == 400


@pytest.mark.gpu
def test_trtllm_run_eval_retrieval():
    model_path = os.getenv('LLAMA3_8B_BASE_TRTLLM')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_TRTLLM to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type tensorrt_llm \
    --output_dir {output_path} \
    --benchmarks math:0 \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/reference \
    ++prompt.few_shot_examples.retrieval_field=question  \
    ++prompt.few_shot_examples.retrieval_file=/code/datasets/math/train_full.jsonl \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++split_name=test \
    ++batch_size=8 \
    ++max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/math/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' not in elem['generation']
        assert elem['error_message'] == '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/math/output-greedy.jsonl"], EVALUATOR_MAP['math']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (10, 15)
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_trtllm_run_labeling():
    model_path = os.getenv('LLAMA3_8B_BASE_TRTLLM')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_TRTLLM to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_labeling.py \
    --model_path {model_path} \
    --server_type tensorrt_llm \
    --output_dir {output_path} \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/base \
    ++prompt.few_shot_examples.examples_type=gsm8k_only_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++dataset=gsm8k \
    ++split_name=train_full \
    ++inference.temperature=0.5 \
    ++skip_filled=False \
    ++batch_size=8 \
    ++max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/output-rs0.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generation']
        assert elem['error_message'] != '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/output-rs0.jsonl"], EVALUATOR_MAP['gsm8k']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (30, 15)
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_nemo_run_eval():
    model_path = os.getenv('LLAMA3_8B_BASE_NEMO')
    if not model_path:
        pytest.skip("Define LLAMA3_8B_BASE_NEMO to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python pipeline/run_eval.py \
    --model_path {model_path} \
    --server_type nemo \
    --output_dir {output_path} \
    --benchmarks gsm8k:0 \
    --num_gpus 1 \
    --num_nodes 1 \
    +prompt=openmathinstruct/base \
    ++prompt.few_shot_examples.examples_type=gsm8k_only_code \
    ++prompt.few_shot_examples.num_few_shots=5 \
    ++prompt.context_type=reference_solution \
    ++split_name=test \
    batch_size=8 \
    max_samples=20 \
"""
    subprocess.run(cmd, shell=True)

    # double checking that code was actually executed
    with open(f"{output_path}/gsm8k/output-greedy.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    for elem in data:
        assert '<llm-code>' in elem['generation']
        assert elem['error_message'] != '<not_executed>'

    # running compute_metrics to check that results are expected
    metrics = compute_metrics([f"{output_path}/gsm8k/output-greedy.jsonl"], EVALUATOR_MAP['gsm8k']())
    assert (int(metrics['sympy_correct']), int(metrics['no_answer'])) == (90, 5)
    assert metrics['num_entries'] == 20
