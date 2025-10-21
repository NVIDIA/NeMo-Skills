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


def eval_qwen3_bfcl(workspace, cluster, expname_prefix, wandb_project):
    model = "Qwen/Qwen3-4B"

    eval(
        ctx=wrap_arguments(
            f"++inference.temperature=0.6 "
            f"++inference.top_p=0.95 "
            f"++inference.tokens_to_generate=8192 "
            f"++model_name={model} "
        ),
        cluster=cluster,
        benchmarks="bfcl_v3",
        model=model,
        server_gpus=2,
        num_jobs=1,
        server_type="vllm",
        output_dir=workspace,
        expname=expname_prefix,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )

    return expname_prefix


def eval_qwen3_online_genselect(workspace, cluster, expname_prefix, wandb_project):
    model = "Qwen/Qwen3-4B"

    output_dir = f"{workspace}/online_genselect/"
    expname = expname_prefix + "_online-genselect"
    eval(
        ctx=wrap_arguments(
            "++inference.temperature=0.6 "
            "++inference.top_p=0.95 "
            "++inference.tokens_to_generate=16384 "
            "++parallel_thinking.mode=genselect "
            "++server.enable_soft_fail=True "
            "++server.context_limit_retry_strategy=reduce_generation "
            "++skip_filled=False "
        ),
        cluster=cluster,
        benchmarks="aime24:1",
        model=model,
        server_gpus=1,
        num_jobs=1,
        server_type="vllm",
        server_args="--async-scheduling --enforce-eager",
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )

    return expname


def eval_qwen3_offline_genselect(workspace, cluster, expname_prefix, wandb_project):
    model = "Qwen/Qwen3-4B"

    output_dir = f"{workspace}/offline_genselect/initial_solutions"
    benchmark = "aime24"
    num_samples = 8

    # Generate initial solutions
    initial_solutions_expname = expname_prefix + "_offline-genselect-initial-solutions"
    eval(
        ctx=wrap_arguments(
            "++inference.temperature=0.6 "
            "++inference.top_p=0.95 "
            "++inference.tokens_to_generate=16384 "  # Generating only 8192 tokens to make it faster
        ),
        cluster=cluster,
        benchmarks=f"{benchmark}:{num_samples}",
        model=model,
        server_gpus=1,
        num_jobs=1,
        server_type="vllm",
        server_args="--async-scheduling --enforce-eager",
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        expname=initial_solutions_expname,
        wandb_project=wandb_project,
        wandb_name=initial_solutions_expname,
    )

    expname = expname_prefix + "_offline-genselect-genselect"
    output_dir = f"{workspace}/offline_genselect/genselect"
    eval(
        ctx=wrap_arguments(
            f"++inference.temperature=0.6 "
            f"++inference.top_p=0.95 "
            f"++inference.tokens_to_generate=8192 "  # Generating only 8192 tokens to make it faster
            f"++parallel_thinking.mode=genselect "
            f"++parallel_thinking.generation_dir={output_dir}/eval-results/{benchmark} "
            f"++server.enable_soft_fail=True "
            f"++server.context_limit_retry_strategy=reduce_generation "
            f"++skip_filled=True "
        ),
        cluster=cluster,
        benchmarks="aime24:1",
        model=model,
        server_gpus=1,
        num_jobs=1,
        server_type="vllm",
        server_args="--async-scheduling --enforce-eager",
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=initial_solutions_expname,
        wandb_project=wandb_project,
        wandb_name=expname,
    )

    return expname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing all experiment data")
    parser.add_argument("--cluster", required=True, help="Cluster name, e.g. oci")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci", help="W&B project name")

    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("bfcl_v3 aime24"))

    # bfcl_expname = eval_qwen3_bfcl(
    #     workspace=args.workspace,
    #     cluster=args.cluster,
    #     expname_prefix=args.expname_prefix,
    #     wandb_project=args.wandb_project,
    # )

    # GenSelect Tests
    online_genselect_expname = eval_qwen3_online_genselect(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    offline_genselect_expname = eval_qwen3_offline_genselect(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    # schedule a dependent check job on the cluster and check if the results are as expected
    checker_cmd = f"python tests/slurm-tests/qwen3_evals/check_results.py --workspace {args.workspace} "

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=[online_genselect_expname, offline_genselect_expname],
    )


if __name__ == "__main__":
    main()
