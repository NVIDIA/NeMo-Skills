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

# needs to define NEMO_SKILLS_TEST_HF_MODEL to run this test
# you'd also need 2+ GPUs to run this test

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[2] / 'pipeline'))
from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, launch_job


@pytest.mark.gpu
def test_hf_trtllm_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f"""cd /code && \
python nemo_skills/conversion/hf_to_trtllm.py \
    --model_dir /model \
    --output_dir /tmp/trtllm \
    --dtype bfloat16 \
    --tp_size 1 \
&& trtllm-build \
    --checkpoint_dir /tmp/trtllm \
    --output_dir /output/trtllm-model \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --context_fmha "enable" \
    --paged_kv_cache "enable" \
    --use_paged_context_fmha "enable" \
    --max_input_len 3584 \
    --max_seq_len 4096 \
    --max_num_tokens 4096 \
    --max_batch_size 8 \
&& cp /model/tokenizer* /output/trtllm-model/
"""

    launch_job(
        cmd,
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=2,
        job_name='test',
        container=CLUSTER_CONFIG["containers"]['tensorrt_llm'],
        mounts=f"{model_path}:/model,{output_path}:/output,{NEMO_SKILLS_CODE}:/code",
    )


@pytest.mark.gpu
def test_hf_nemo_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    cmd = f"""cd /code && \
HF_TOKEN={os.environ['HF_TOKEN']} python nemo_skills/conversion/hf_to_nemo.py \
    --in-path /model \
    --out-path /output/model.nemo \
    --hf-model-name meta-llama/Meta-Llama-3-8B \
    --precision 16
"""

    launch_job(
        cmd,
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=2,
        job_name='test',
        container=CLUSTER_CONFIG["containers"]['nemo'],
        mounts=f"{model_path}:/model,{output_path}:/output,{NEMO_SKILLS_CODE}:/code",
    )


@pytest.mark.gpu
def test_nemo_hf_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    output_path = os.getenv('NEMO_SKILLS_TEST_OUTPUT', '/tmp')

    # there is a bug in transformers related to slurm, so unsetting the vars
    # TODO: remove this once the bug is fixed
    cmd = f"""cd /code && unset SLURM_PROCID && unset SLURM_LOCALID && \
HF_TOKEN={os.environ['HF_TOKEN']} python nemo_skills/conversion/nemo_to_hf.py \
    --in-path /model \
    --out-path /output/hf-model \
    --hf-model-name meta-llama/Meta-Llama-3-8B \
    --precision 16 \
    --max-shard-size 10GB
"""

    launch_job(
        cmd,
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=2,
        job_name='test',
        container=CLUSTER_CONFIG["containers"]['nemo'],
        mounts=f"{model_path}:/model,{output_path}:/output,{NEMO_SKILLS_CODE}:/code",
    )
