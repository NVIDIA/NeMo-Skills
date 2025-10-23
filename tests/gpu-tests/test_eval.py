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
import subprocess
from importlib import import_module
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import eval as cli_eval
from nemo_skills.pipeline.cli import prepare_data, wrap_arguments
from tests.conftest import docker_rm


@pytest.mark.gpu
def test_trtllm_eval():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/trtllm-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args='--backend pytorch' "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]

    # rough check, since exact accuracy varies depending on gpu type
    if model_type == "qwen":
        assert metrics["symbolic_correct"] >= 70

    assert metrics["num_entries"] == 10


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["trtllm"])
def test_trtllm_code_execution_eval(server_type):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    # we are using the base prompt for Qwen to make it follow few-shots
    if model_type == "qwen":
        # tokenizer = "Qwen/Qwen3-1.7B"
        code_tags = "qwen"
    else:
        raise ValueError("Only qwen models are supported in this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --with_sandbox "
        f"    ++stop_phrase='\\n\\n\\n\\n\\n\\n' "
        f"    --server_args='--backend pytorch' "
        f"    ++tokenizer=Qwen/Qwen3-1.7B-Base "
        f"    ++code_tags={code_tags} "
        f"    ++examples_type=gsm8k_text_with_code "
        f"    ++max_samples=20 "
        f"    ++code_execution=True "
        f"    ++inference.tokens_to_generate=2048 "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    if model_type == "qwen":
        assert metrics["symbolic_correct"] >= 70
    assert metrics["num_entries"] == 20


@pytest.mark.gpu
@pytest.mark.parametrize(
    "server_type,server_args",
    [
        ("vllm", "--enforce-eager --max-model-len 4096"),
        ("sglang", "--context-length 4096"),
        ("trtllm", "--backend pytorch"),
    ],
)
def test_hf_eval(server_type, server_args):
    # this test expects qwen3-1.7b to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/ifeval will be cut to 164 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    if model_type != "qwen":
        raise ValueError(f"Only running this test for qwen models, got {model_type}")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks algebra222,human-eval,ifeval,mmlu "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_jobs 1 "
        f"    --server_args='{server_args}' "
        f"    ++max_samples=164 "
        f"    ++inference.tokens_to_generate=2048 "
        f"    ++parse_reasoning=True "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {output_dir}",
        shell=True,
        check=True,
    )

    with open(f"{output_dir}/eval-results/algebra222/metrics.json", "r") as f:
        metrics = json.load(f)["algebra222"]["pass@1"]

    assert metrics["symbolic_correct"] >= 75
    assert metrics["num_entries"] == 164

    with open(f"{output_dir}/eval-results/human-eval/metrics.json", "r") as f:
        metrics = json.load(f)["human-eval"]["pass@1"]

    assert metrics["passing_base_tests"] >= 35
    assert metrics["passing_plus_tests"] >= 35
    assert metrics["num_entries"] == 164

    with open(f"{output_dir}/eval-results/ifeval/metrics.json", "r") as f:
        metrics = json.load(f)["ifeval"]["pass@1"]

    assert metrics["prompt_strict_accuracy"] >= 60
    assert metrics["instruction_strict_accuracy"] >= 60
    assert metrics["prompt_loose_accuracy"] >= 60
    assert metrics["instruction_loose_accuracy"] >= 60
    assert metrics["num_prompts"] == 164

    with open(f"{output_dir}/eval-results/mmlu/metrics.json", "r") as f:
        metrics = json.load(f)["mmlu"]["pass@1"]
    assert metrics["symbolic_correct"] >= 60
    assert metrics["num_entries"] == 164


@pytest.mark.gpu
def test_megatron_eval():
    try:
        model_path = require_env_var("NEMO_SKILLS_TEST_MEGATRON_MODEL")
        model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")
    except ValueError:
        pytest.skip("Define NEMO_SKILLS_TEST_MEGATRON_MODEL and NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    if model_type != "qwen":
        raise ValueError(f"Only running this test for qwen models, got {model_type}")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/megatron-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type megatron "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++max_samples=5 "
        f"    ++tokenizer=Qwen/Qwen3-1.7B "
        f"    --server_args='--tokenizer-model Qwen/Qwen3-1.7B --inference-max-requests=20' "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    # TODO: something is broken in megatron inference here as this should be 50!
    assert metrics["symbolic_correct"] >= 40
    assert metrics["num_entries"] == 5


@pytest.mark.gpu
def test_prepare_and_eval_all_datasets():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    config_dir = Path(__file__).absolute().parent
    datasets_dir = Path(__file__).absolute().parents[2] / "nemo_skills" / "dataset"
    excluded_datasets = {"__pycache__", "ruler", "human-eval", "mbpp"}

    dataset_names = sorted(
        dataset.name
        for dataset in datasets_dir.iterdir()
        if dataset.is_dir() and (dataset / "prepare.py").exists() and dataset.name not in excluded_datasets
    )

    assert dataset_names, "No datasets found to prepare and evaluate"

    judge_datasets = []
    for dataset in dataset_names:
        dataset_module = import_module(f"nemo_skills.dataset.{dataset}")
        if getattr(dataset_module, "JUDGE_PIPELINE_ARGS", None):
            judge_datasets.append(dataset)

    non_judge_datasets = [dataset for dataset in dataset_names if dataset not in judge_datasets]

    data_dir = Path(f"/tmp/nemo-skills-tests/{model_type}/data")
    docker_rm([str(data_dir)])

    prepare_data(
        ctx=wrap_arguments(" ".join(dataset_names)),
        cluster="test-local",
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        expname=f"prepare-all-datasets-{model_type}",
    )

    eval_kwargs = dict(
        cluster="test-local",
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        model=model_path,
        server_type="sglang",
        server_gpus=1,
        server_nodes=1,
    )

    collected_datasets = set()
    common_ctx = "++max_samples=2 ++inference.tokens_to_generate=100"

    if non_judge_datasets:
        output_dir = f"/tmp/nemo-skills-tests/{model_type}/all-datasets-eval"
        docker_rm([output_dir])
        cli_eval(
            ctx=wrap_arguments(common_ctx),
            output_dir=output_dir,
            benchmarks=",".join(non_judge_datasets),
            expname=f"eval-all-datasets-{model_type}",
            **eval_kwargs,
        )

        eval_results_dir = Path(output_dir) / "eval-results"
        for dataset in non_judge_datasets:
            metrics_path = eval_results_dir / dataset / "metrics.json"
            assert metrics_path.exists(), f"Missing metrics for {dataset}"
            collected_datasets.add(dataset)

    if judge_datasets:
        judge_output_dir = f"/tmp/nemo-skills-tests/{model_type}/all-datasets-eval-judge"
        docker_rm([judge_output_dir])
        cli_eval(
            ctx=wrap_arguments(common_ctx),
            output_dir=judge_output_dir,
            benchmarks=",".join(judge_datasets),
            expname=f"eval-all-datasets-judge-{model_type}",
            judge_model=model_path,
            judge_server_type="sglang",
            judge_server_gpus=1,
            **eval_kwargs,
        )

        judge_results_dir = Path(judge_output_dir) / "eval-results"
        for dataset in judge_datasets:
            metrics_path = judge_results_dir / dataset / "metrics.json"
            assert metrics_path.exists(), f"Missing metrics for judge benchmark {dataset}"
            collected_datasets.add(dataset)

    assert set(dataset_names) == collected_datasets
