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
import subprocess
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_hf_trtllm_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")

    cmd = (
        f"python -m nemo_skills.pipeline.convert "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --input_model {model_path} "
        f"    --output_model /tmp/nemo-skills-tests/conversion/hf-to-trtllm/model "
        f"    --convert_from hf "
        f"    --convert_to trtllm "
        f"    --num_gpus 1 "
        f"    --hf_model_name meta-llama/Meta-Llama-3.1-8B "
    )

    subprocess.run(cmd, shell=True, check=True)


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
