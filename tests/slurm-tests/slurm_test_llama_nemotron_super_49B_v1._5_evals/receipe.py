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
    # step1: local data preparation
    prepare_datasets(["aime24", "aime25"])

    # step2: download models
    cmd = (
        f"huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 --local-dir {workspace}/Llama-3_3-Nemotron-Super-49B-v1_5 && "
        f"huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir {workspace}/Qwen2.5-32B-Instruct"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-models",
        log_dir=f"{workspace}/download-assets",
    )

    # Update config to support 128k
    cmd = (
        f'jq \'. + {{"rope_scaling": {{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}}}\' '
        f"{workspace}/Qwen2.5-32B-Instruct/config.json > {workspace}/Qwen2.5-32B-Instruct/config_tmp.json && "
        f"mv {workspace}/Qwen2.5-32B-Instruct/config_tmp.json {workspace}/Qwen2.5-32B-Instruct/config.json"
    )

    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-patch-qwen-config",
        log_dir=f"{workspace}/download-assets",
        run_after=f"{expname_prefix}-download-models",
        container="nemo",
    )

    # step3: prepare ruler data on local cluster
    ruler_cmd = [
        "ns",
        "prepare_data",
        "--cluster",
        cluster,
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
        f"{expname_prefix}-patch-qwen-config",
        "--expname",
        f"{expname_prefix}-download-ruler-data",
    ]
    subprocess.run(ruler_cmd, check=True)


def eval_reasoning_on(workspace, cluster, expname_prefix):
    """
    Run evals in Reasoning ON mode
    """
    base_model = f"{workspace}/Llama-3_3-Nemotron-Super-49B-v1_5"

    # Common settings for reasoning ON
    common_infer = [
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++skip_filled=True",
        "++system_message=",
    ]

    # Math / Code / Science (Reasoning ON)
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
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on",
        "--benchmarks",
        "gpqa:16,scicode:16,math-500:16,aime24:16,aime25:16",
        "--server_gpus=8",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-math-code-science-on",
    ]
    subprocess.run(cmd_reasoning, check=True)

    # MMLU needs two continous jobs run on A100 GPU
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
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on",
        "--benchmarks",
        "mmlu-pro:16",
        "--server_gpus=8",
        "--dependent_jobs=1",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-math-code-science-on",
    ]
    subprocess.run(cmd_reasoning, check=True)

    # LiveCodeBench (Reasoning ON)
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
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on",
        "--benchmarks",
        "livecodebench:16",
        "--split",
        "test_v5_2410_2502",
        "--server_gpus=8",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-livecode-on",
    ]
    subprocess.run(cmd_livecode, check=True)

    # HLE (Reasoning ON)
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
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on",
        "--benchmarks",
        "hle:16",
        "--server_gpus=8",
        "--dependent_jobs=1",
        "--judge_model",
        f"{workspace}/Qwen2.5-32B-Instruct",
        "--judge_server_type=vllm",
        "--judge_server_gpus=8",
        "--extra_judge_args=++inference.tokens_to_generate=4096 ",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-hle-on",
    ]
    subprocess.run(cmd_hle, check=True)

    # BFCL (Reasoning ON)
    cmd_bfcl = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--benchmarks",
        "bfcl_v3",
        "--model",
        base_model,
        "--server_gpus=8",
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on_tool_calling",
        *common_infer,
        "++use_client_parsing=False",
        '--server_args=--tool-parser-plugin "{}" --tool-call-parser llama_nemotron_json --enable-auto-tool-choice'.format(
            f"{base_model}/llama_nemotron_toolcall_parser_no_streaming.py"
        ),
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-bfcl-on",
    ]
    subprocess.run(cmd_bfcl, check=True)

    # RULER (Reasoning ON)  Note: no tokens_to_generate
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
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_on_ruler",
        "--benchmarks",
        "ruler.nemotron_super_128k",
        "--data_dir",
        f"{workspace}/ns-data",
        "--server_gpus=8",
        "--dependent_jobs=1",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++skip_filled=True",
        "++system_message=",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-ruler-on",
    ]
    subprocess.run(cmd_ruler, check=True)


def eval_reasoning_off(workspace, cluster, expname_prefix):
    """
    Run evals in Reasoning OFF mode
    temperature=0.0, top_p=1.0, system_message=/no_think
    Keep tokens_to_generate=65536 (except RULER)
    """
    base_model = f"{workspace}/Llama-3_3-Nemotron-Super-49B-v1_5"

    # Common settings for reasoning OFF
    common_infer = [
        "++inference.tokens_to_generate=65536",
        "++inference.temperature=0.0",
        "++inference.top_p=1.0",
        "++skip_filled=True",
        "++system_message=/no_think",
    ]

    # Math / Code / Science (Reasoning OFF)
    cmd_reasoning_off = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_off",
        "--benchmarks",
        "gpqa:16,mmlu-pro:16,scicode:16,math-500:16,aime24:16,aime25:16",
        "--server_gpus=8",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-math-code-science-off",
    ]
    subprocess.run(cmd_reasoning_off, check=True)

    # LiveCodeBench (Reasoning OFF)
    cmd_livecode_off = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_off",
        "--benchmarks",
        "livecodebench:16",
        "--split",
        "test_v5_2410_2502",
        "--server_gpus=8",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-livecode-off",
    ]
    subprocess.run(cmd_livecode_off, check=True)

    # HLE (Reasoning OFF)
    cmd_hle_off = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_off",
        "--benchmarks",
        "hle:16",
        "--server_gpus=8",
        "--dependent_jobs=1",
        "--judge_model",
        f"{workspace}/Qwen2.5-32B-Instruct",
        "--judge_server_type=vllm",
        "--judge_server_gpus=8",
        "--extra_judge_args=++inference.tokens_to_generate=4096 ",
        *common_infer,
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-hle-off",
    ]
    subprocess.run(cmd_hle_off, check=True)

    # BFCL (Reasoning OFF)
    cmd_bfcl_off = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--benchmarks",
        "bfcl_v3",
        "--model",
        base_model,
        "--server_gpus=8",
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_off_tool_calling",
        *common_infer,
        "++use_client_parsing=False",
        '--server_args=--tool-parser-plugin "{}" --tool-call-parser llama_nemotron_json --enable-auto-tool-choice'.format(
            f"{base_model}/llama_nemotron_toolcall_parser_no_streaming.py"
        ),
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-bfcl-off",
    ]
    subprocess.run(cmd_bfcl_off, check=True)

    # RULER (Reasoning OFF)  Note: no tokens_to_generate
    cmd_ruler_off = [
        "ns",
        "eval",
        "--cluster",
        cluster,
        "--model",
        base_model,
        "--server_type",
        "vllm",
        "--output_dir",
        f"{workspace}/llama_nemotron_49b_1_5_reasoning_off_ruler",
        "--benchmarks",
        "ruler.nemotron_super_128k",
        "--data_dir",
        f"{workspace}/ns-data",
        "--server_gpus=8",
        "--dependent_jobs=1",
        "++inference.temperature=0.0",
        "++inference.top_p=1.0",
        "++skip_filled=True",
        "++system_message=/no_think",
        "--run_after",
        f"{expname_prefix}-download-ruler-data",
        "--expname",
        f"{expname_prefix}-ruler-off",
    ]
    subprocess.run(cmd_ruler_off, check=True)


# Prepare evaluation data locally first
def prepare_data_locally():
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
    subprocess.run(cmd, capture_output=False)


def main():
    parser = argparse.ArgumentParser(description="Run Nemotron eval pipeline")
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name, e.g. oci")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")

    args = parser.parse_args()

    # launch for eval jobs
    prepare_data_locally()
    download_models_ruler_data(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)
    eval_reasoning_on(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)
    eval_reasoning_off(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)

    # schedule a dependent check job on the cluster and check if the results are as expected

    checker = (
        f"cd /nemo_run/code/tests/slurm-tests/slurm_test_llama_nemotron_super_49B_v1._5_evals && "
        f"python check.py --workspace {args.workspace} "
    )

    run_cmd(
        ctx=wrap_arguments(checker),
        cluster=args.cluster,
        expname=f"check-eval-results-for-llama-49b",
        log_dir=f"{args.workspace}/logs",
        run_after=[
            f"{args.expname_prefix}-math-code-science-on",
            f"{args.expname_prefix}-livecode-on",
            f"{args.expname_prefix}-hle-on",
            f"{args.expname_prefix}-bfcl-on",
            f"{args.expname_prefix}-ruler-on",
            f"{args.expname_prefix}-math-code-science-off",
            f"{args.expname_prefix}-livecode-off",
            f"{args.expname_prefix}-hle-off",
            f"{args.expname_prefix}-bfcl-off",
            f"{args.expname_prefix}-ruler-off",
        ],
    )


if __name__ == "__main__":
    main()
