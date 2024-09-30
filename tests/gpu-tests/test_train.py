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
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from nemo_skills.evaluation.metrics import compute_metrics
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import eval, train


@pytest.mark.gpu
def test_sft():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    train(
        ctx=wrap_arguments(
            "++trainer.sft.val_check_interval=7 "
            "++trainer.sft.save_interval=7 "
            "++trainer.sft.limit_val_batches=1 "
            "++trainer.sft.max_steps=15 "
            "++trainer.sft.max_epochs=10 "
            "++model.data.train_ds.add_eos=False "
            "++model.data.train_ds.global_batch_size=10 "
            "++model.data.train_ds.micro_batch_size=2 "
            "++model.optim.lr=1e-6 "
            "++model.optim.sched.warmup_steps=0 "
            "++model.tensor_model_parallel_size=1 "
            "++model.pipeline_model_parallel_size=1 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-sft",
        output_dir="/tmp/nemo-skills-tests/test-sft",
        nemo_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-sft-data.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments("++prompt_template=llama3-instruct " "++split=test " "++batch_size=8 " "++max_samples=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model="/tmp/nemo-skills-tests/test-sft/model-averaged-nemo",
        server_type="nemo",
        output_dir="/tmp/nemo-skills-tests/test-sft/evaluation",
        benchmarks="gsm8k:0",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        partition="interactive",
    )

    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/test-sft/evaluation/eval-results/gsm8k/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS(),
    )
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10


@pytest.mark.gpu
def test_dpo():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")

    train(
        ctx=wrap_arguments(
            "++trainer.dpo.val_check_interval=1 "
            "++trainer.dpo.save_interval=1 "
            "++trainer.dpo.limit_val_batches=1 "
            "++trainer.dpo.max_steps=3 "
            "++trainer.dpo.max_epochs=10 "
            "++model.data.train_ds.add_eos=False "
            "++model.global_batch_size=1 "
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
        output_dir="/tmp/nemo-skills-tests/test-dpo",
        nemo_model=model_path,
        num_nodes=1,
        num_gpus=1,
        num_training_jobs=1,
        training_data="/nemo_run/code/tests/data/small-dpo-data.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments("++prompt_template=llama3-instruct " "++split=test " "++batch_size=8 " "++max_samples=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model="/tmp/nemo-skills-tests/test-dpo/model-averaged-nemo",
        server_type="nemo",
        output_dir="/tmp/nemo-skills-tests/test-dpo/evaluation",
        benchmarks="gsm8k:0",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        partition="interactive",
    )

    metrics = compute_metrics(
        [f"/tmp/nemo-skills-tests/test-dpo/evaluation/eval-results/gsm8k/output-greedy.jsonl"],
        importlib.import_module('nemo_skills.dataset.gsm8k').METRICS_CLASS(),
    )
    # only checking the total, since model is tiny
    assert metrics['num_entries'] == 10
