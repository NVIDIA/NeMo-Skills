# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

# Download model and ruler on cluster
import subprocess

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.pipeline.cli import convert, eval, generate, run_cmd, sft_nemo_rl, train, wrap_arguments


def download_models_ruler_data(workspace, cluster, expname_prefix):
    # data preparation needs to run locally without container, so not wrapping with run_cmd
    prepare_datasets(["aime24", "aime25"])

    # download the models
    cmd = f"huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 --local-dir {workspace}/Llama-3_3-Nemotron-Super-49B-v1_5 "
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-models",
        log_dir=f"{workspace}/download-assets",
    )

    # Prepare ruler data on local cluster using subprocess
    ruler_cmd = [
        "ns",
        "prepare_data",
        "--cluster",
        f"{cluster}",
        "ruler",
        "--setup",
        "nemotron_super_128k",
        "--tokenizer_path",
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        "--max_seq_length",
        "131072",
        "--data_dir",
        f"{workspace}/ns-data",
        "--run_after",
        f"{expname_prefix}-download-models",
        "--expname",
        f"{expname_prefix}-download-ruler-data",
    ]

    subprocess.run(ruler_cmd, check=True, capture_output=False)


def eval(workspace, cluster, expname_prefix):
    base_model = f"{workspace}/Llama-3_3-Nemotron-Super-49B-v1_5"

    # ========== Math / Code / Science Reasoning ==========
    cmd_reasoning = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5",
        "--benchmarks",
        "gpqa:16,mmlu-pro:16,scicode:16,math-500:16,aime24:16,aime25:16",
        "--server_gpus=2",
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++system_message=",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-math-code-science-reasoning-on",
    ]
    subprocess.run(cmd_reasoning, check=True)

    # ========== LiveCodeBench ==========
    cmd_livecode = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5",
        "--benchmarks",
        "livecodebench:16",
        "--split",
        "test_v5_2410_2502",
        "--server_gpus=2",
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++system_message=",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-livecode-reasoning-on",
    ]
    subprocess.run(cmd_livecode, check=True)

    # ========== HLE ==========
    cmd_hle = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5",
        "--benchmarks",
        "hle:16",
        "--server_gpus=2",
        "--judge_model=o3-mini-20250131",
        "--extra_judge_args=++inference.tokens_to_generate=4096 ++max_concurrent_requests=8",
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++system_message=",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-hle-reasoning-on",
    ]
    subprocess.run(cmd_hle, check=True)

    # ========== BFCL ==========
    cmd_bfcl = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--benchmarks",
        "bfcl_v3",
        "--model",
        base_model,
        "--server_gpus=2",
        "--server_type=vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_tool_calling",
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++system_message=",
        "++use_client_parsing=False",
        "--server_args=--tool-parser-plugin "
        f'"{base_model}/llama_nemotron_toolcall_parser_no_streaming.py" '
        "--tool-call-parser llama_nemotron_json "
        "--enable-auto-tool-choice",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-bfcl-reasoning-on",
    ]
    subprocess.run(cmd_bfcl, check=True)

    # ========== RULER ==========
    cmd_ruler = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_ruler",
        "--benchmarks",
        "ruler.nemotron_super_128k",
        "--data_dir",
        f"{workspace}/ns-data",
        "--server_gpus=2",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++system_message=",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-ruler-reasoning-on",
    ]
    subprocess.run(cmd_ruler, check=True)


# Prepare evaluation data locally first
cmd = [
    "ns",
    "prepare_data",
    "gpqa",
    "mmlu-pro",
    "hle",
    "livecodebench",
    "scicode",
    "bfcl_v3",
    "math-500",
    "aime24",
    "aime25",
]
# subprocess.run(cmd, capture_output=False)


workspace = "/lustre/fsw/portfolios/llmservice/users/wedu/llm/test-49b"
cluster = "oci"
expname_prefix = "test-49b"
eval(workspace=workspace, cluster=cluster, expname_prefix=expname_prefix)
