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

sys.path.append(str(Path(__file__).absolute().parents[2] / 'pipeline'))
from compute_metrics import compute_metrics


def test_sft_pipeline():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f""" \
python {Path(__file__).absolute().parents[2]}/datasets/gsm8k/prepare.py --split_name validation && \
export NEMO_SKILLS_DATA={Path(__file__).absolute().parents[2] / 'datasets'} && \
export NEMO_SKILLS_RESULTS={output_path} && \
python pipeline/run_pipeline.py \
      --expname test \
      --nemo_model {model_path} \
      --num_nodes 1 \
      --num_gpus 2 \
      --disable_wandb \
      --extra_eval_args "++max_samples=4 --benchmarks gsm8k:1 math:0 --num_jobs 1 --num_gpus 1" \
      ++model.data.train_ds.file_path=/data/gsm8k/validation-sft.jsonl \
      ++trainer.sft.max_steps=15 \
      ++trainer.sft.val_check_interval=10 \
      ++trainer.sft.limit_val_batches=2 \
      ++model.data.train_ds.global_batch_size=4 \
      ++model.tensor_model_parallel_size=2 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.optim.lr=1e-6 \
"""
    subprocess.run(cmd, shell=True)

    # only checking the total, since model is tiny
    for gen_file in ['gsm8k/output-greedy.jsonl', 'gsm8k/output-rs0.jsonl', 'math/output-greedy.jsonl']:
        *_, total = compute_metrics([f"{output_path}/nemo-skills-exps/results/test/{gen_file}"])
        assert total == 4
