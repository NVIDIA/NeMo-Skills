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
import sys
from pathlib import Path

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.pipeline.cli import eval, generate, sft_nemo_rl, train, wrap_arguments
from tests.conftest import docker_rm


@pytest.mark.gpu
def test_sft_nemo_rl():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-nemo-rl"

    # need to clean up current cluster configuration as we mount /tmp and it causes problems
    docker_rm(['/tmp/ray/ray_current_cluster', output_dir])

    sft_nemo_rl(
        ctx=wrap_arguments(
            '++sft.max_num_steps=5 '
            '++policy.dtensor_cfg.tensor_parallel_size=1 '
            '++checkpointing.save_period=2 '
            '++policy.train_global_batch_size=2 '
            '++policy.train_micro_batch_size=1 '
            '++policy.optimizer.kwargs.lr=1e-6 '
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-sft-nemo-rl",
        output_dir=output_dir,
        hf_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-sft-data.test",
        disable_wandb=True,
        cache_dir="/tmp/nemo-skills-tests/nemo-rl-cache",
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments(
            f"++prompt_template={prompt_template} ++split=test ++max_samples=10 ++inference.tokens_to_generate=10"
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/final_hf_model",
        server_type="vllm",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k:0",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
    )

    metrics = ComputeMetrics(benchmark='gsm8k').compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["all"]["greedy"]
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10


@pytest.mark.gpu
def test_sft_aligner():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-aligner"
    docker_rm([output_dir])

    train(
        ctx=wrap_arguments(
            "++trainer.sft.save_interval=2 "
            "++trainer.sft.limit_val_batches=1 "
            "++trainer.sft.max_steps=5 "
            "++trainer.sft.max_epochs=10 "
            "++model.data.train_ds.add_eos=False "
            "++model.data.train_ds.global_batch_size=2 "
            "++model.data.train_ds.micro_batch_size=1 "
            "++model.optim.lr=1e-6 "
            "++model.optim.sched.warmup_steps=0 "
            "++model.tensor_model_parallel_size=1 "
            "++model.pipeline_model_parallel_size=1 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-sft",
        output_dir=output_dir,
        nemo_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-sft-data.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments(f"++prompt_template={prompt_template} ++split=test ++batch_size=8 ++max_samples=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/model-averaged-nemo",
        server_type="nemo",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k:0",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        partition="interactive",
    )

    metrics = ComputeMetrics(benchmark='gsm8k').compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["all"]["greedy"]
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10


@pytest.mark.gpu
def test_dpo_aligner():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-dpo-aligner"
    docker_rm([output_dir])

    train(
        ctx=wrap_arguments(
            "++trainer.dpo.val_check_interval=1 "
            "++trainer.dpo.save_interval=1 "
            "++trainer.dpo.limit_val_batches=1 "
            "++trainer.dpo.max_steps=3 "
            "++trainer.dpo.max_epochs=10 "
            "++model.data.train_ds.add_eos=False "
            "++model.global_batch_size=2 "
            "++model.micro_batch_size=1 "
            "++model.optim.lr=1e-6 "
            "++model.optim.sched.warmup_steps=0 "
            "++model.tensor_model_parallel_size=1 "
            "++model.pipeline_model_parallel_size=1 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-dpo",
        training_algo="dpo",
        output_dir=output_dir,
        nemo_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-dpo-data.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments(f"++prompt_template={prompt_template} ++split=test ++batch_size=8 ++max_samples=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/model-averaged-nemo",
        server_type="nemo",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k:0",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        partition="interactive",
    )

    metrics = ComputeMetrics(benchmark='gsm8k').compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["all"]["greedy"]
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10


@pytest.mark.gpu
@pytest.mark.parametrize("test_mode", ["unit", "integration"])
def test_rm_aligner(test_mode):
    seeds_supported_models = ['llama']
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    if test_mode == "unit":
        input_dir_seeds = "/nemo_run/code/tests/data/score_rm_inputs"
        input_dir_greedy = "/nemo_run/code/tests/data/score_rm_inputs"
        expected_scores_per_file = 5
    else:  # test_mode == "integration"
        input_dir_greedy = os.getenv('NEMO_SKILLS_TEST_RM_INPUTS_GREEDY')
        if not input_dir_greedy:
            pytest.skip("Define NEMO_SKILLS_TEST_RM_INPUTS_SEEDS_GREEDY to run this test")
        if model_type in seeds_supported_models:
            input_dir_seeds = os.getenv('NEMO_SKILLS_TEST_RM_INPUTS_SEEDS')
            if not input_dir_seeds:
                pytest.skip("Define NEMO_SKILLS_TEST_RM_INPUTS_SEEDS to run this test")
        expected_scores_per_file = int(os.getenv('NEMO_SKILLS_TEST_RM_EXPECTED_SCORES_PER_FILE'))
        if not expected_scores_per_file:
            pytest.skip("Define NEMO_SKILLS_TEST_RM_EXPECTED_SCORES_PER_FILE to run this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-rm-aligner"
    docker_rm([output_dir])

    train(
        ctx=wrap_arguments(
            "++trainer.rm.val_check_interval=1 "
            "++trainer.rm.save_interval=1 "
            "++trainer.rm.limit_val_batches=1 "
            "++trainer.rm.max_steps=3 "
            "++trainer.rm.max_epochs=10 "
            "++model.data.train_ds.add_eos=False "
            "++model.global_batch_size=2 "
            "++model.micro_batch_size=1 "
            "++model.optim.lr=1e-6 "
            "++model.optim.sched.warmup_steps=0 "
            "++model.tensor_model_parallel_size=1 "
            "++model.pipeline_model_parallel_size=1 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-rm",
        training_algo="rm",
        output_dir=output_dir,
        nemo_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-rm-data.test",
        disable_wandb=True,
    )

    assert os.path.exists(f"{output_dir}/model-averaged-nemo")

    generate(
        ctx=wrap_arguments(
            f"++batch_size=2 "
            f"++input_dir={input_dir_greedy} "
            f"++prompt_config=generic/math-base "
            f"++prompt_template=llama3-base "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        output_dir=f"{output_dir}/score",
        server_type="nemo",
        generation_type="reward",
        expname="test-rm",
        model=f"{output_dir}/model-averaged-nemo",
        server_gpus=1,
        server_nodes=1,
        partition="interactive",
        num_random_seeds=None,
    )

    assert os.path.exists(f"{output_dir}/score/output.jsonl")
    rm_output = [json.loads(line) for line in open(f"{output_dir}/score/output.jsonl")]
    assert len(rm_output) == expected_scores_per_file
    assert all("reward_model_score" in line for line in rm_output)

    if model_type in seeds_supported_models:
        generate(
            ctx=wrap_arguments(
                f"++batch_size=2 "
                f"++input_dir={input_dir_seeds} "
                f"++prompt_config=generic/math-base "
                f"++prompt_template=llama3-base "
            ),
            cluster="test-local",
            config_dir=Path(__file__).absolute().parent,
            output_dir=f"{output_dir}/score",
            server_type="nemo",
            generation_type="reward",
            expname="test-rm",
            model=f"{output_dir}/model-averaged-nemo",
            server_gpus=1,
            server_nodes=1,
            partition="interactive",
            num_random_seeds=3,
        )

        for rs in range(3):
            assert os.path.exists(f"{output_dir}/score/output-rs{rs}.jsonl")
            rm_output = [json.loads(line) for line in open(f"{output_dir}/score/output-rs{rs}.jsonl")]
            assert len(rm_output) == expected_scores_per_file
            assert all("reward_model_score" in line for line in rm_output)
