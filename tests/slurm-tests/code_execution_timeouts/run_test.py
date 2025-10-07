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


def eval_gpt_oss_hmmt(workspace: str, cluster: str, expname_prefix: str, wandb_project: str | None) -> str:
    """Submit code execution timeouts evaluation."""

    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++max_concurrent_requests=1024 "
            "++prompt_config=gpt-oss/math "
            "++code_tags=gpt-oss "
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++server.code_execution.code_execution_timeout=120 "
            "++use_completions_api=true "
            "++chat_template_kwargs.reasoning_effort=high "
            "++chat_template_kwargs.builtin_tools=[python] "
        ),
        cluster=cluster,
        benchmarks="hmmt_feb25:16",
        model="openai/gpt-oss-120b",
        server_gpus=8,
        num_jobs=1,
        server_type="vllm",
        output_dir=workspace,
        server_args="--async-scheduling",
        with_sandbox=True,
        expname=expname_prefix,
        wandb_project=wandb_project,
        wandb_name=expname_prefix,
    )

    return expname_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--expname_prefix", required=True)
    parser.add_argument("--wandb_project", default="nemo-skills-slurm-ci")
    args = parser.parse_args()

    prepare_data(ctx=wrap_arguments("hmmt_feb25"))

    eval_expname = eval_gpt_oss_hmmt(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
        wandb_project=args.wandb_project,
    )

    checker_cmd = (
        "python tests/slurm-tests/code_execution_timeouts/check_results.py " + f"--workspace {args.workspace}"
    )

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=eval_expname,
    )


if __name__ == "__main__":
    main()
