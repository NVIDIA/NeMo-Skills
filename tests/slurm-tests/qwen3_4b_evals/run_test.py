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

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments


def setup(workspace, cluster, expname_prefix):
    # download models
    model = "Qwen/Qwen3-4B"
    cmd = f"huggingface-cli download {model}"
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"{expname_prefix}-download-models",
        log_dir=f"{workspace}/download-assets",
    )


def eval_reasoning_on(workspace, cluster, expname_prefix, wandb_project):
    """Run evals in Reasoning ON mode"""
    base_model = "Qwen/Qwen3-4B"

    # Common settings for reasoning ON
    common_params = "++inference.temperature=0.6 ++inference.top_p=0.95 "
    tokens_to_generate = "++inference.tokens_to_generate=8192 "

    # BFCL (Reasoning ON)
    eval(
        ctx=wrap_arguments(f"{common_params} {tokens_to_generate}"),
        cluster=cluster,
        benchmarks="bfcl_v3",
        model=base_model,
        server_gpus=8,
        num_jobs=1,
        server_type="vllm",
        output_dir=f"{workspace}/reasoning_on_tool_calling",
        expname=f"{expname_prefix}-bfcl-on",
        run_after=f"{expname_prefix}-download-models",
        wandb_project=wandb_project,
        wandb_name=f"{expname_prefix}-qwen3-4b-eval-reasoning-on",
    )

    return [
        f"{expname_prefix}-bfcl-on",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Nemotron eval pipeline")
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name, e.g. oci")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("bfcl_v3"))

    setup(workspace=args.workspace, cluster=args.cluster, expname_prefix=args.expname_prefix)

    reasoning_on_expnames = eval_reasoning_on(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    # schedule a dependent check job on the cluster and check if the results are as expected

    checker = (
        f"cd /nemo_run/code/tests/slurm-tests/qwen3_4b_evals && python check_results.py --workspace {args.workspace} "
    )

    run_cmd(
        ctx=wrap_arguments(checker),
        cluster=args.cluster,
        expname="check-eval-results-for-qwen3-4b",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=reasoning_on_expnames,
    )


if __name__ == "__main__":
    main()
