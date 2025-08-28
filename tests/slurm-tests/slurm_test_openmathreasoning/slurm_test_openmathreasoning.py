# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import subprocess
import sys

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def main():
    ap = argparse.ArgumentParser(
        description="Run OpenMathReasoning and then schedule an on-cluster SFT results check."
    )
    ap.add_argument("--cluster", required=True, help="Cluster name (e.g., oci, local)")
    ap.add_argument("--backend", required=True, choices=["nemo-aligner", "nemo-rl"], help="Training backend")
    ap.add_argument("--workspace", required=True, help="Workspace path")
    ap.add_argument("--wandb_project", default=None, help="W&B project name")
    ap.add_argument("--expname_prefix", required=True, help="Experiment name prefix used inside the recipe")
    ap.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging in the recipe")
    args = ap.parse_args()

    # 1) Run simplified_recipe
    def run_simplified_recipe(args):
        cmd = f"""
        python -m recipes.openmathreasoning.scripts.simplified_recipe \
            --cluster {args.cluster} \
            --workspace {args.workspace} \
            --training_backend {args.backend} \
            --expname_prefix {args.expname_prefix}
        """

        if args.disable_wandb:
            cmd += " --disable_wandb"
        elif args.wandb_project:
            cmd += f" --wandb_project {args.wandb_project}"

        subprocess.run(cmd, shell=True, check=True)

    run_simplified_recipe(args)

    # 2) Schedule a dependent check job ON THE CLUSTER.
    #    This downloads the checker into the workspace and runs it there.

    checker = (
        f"cd /nemo_run/code/tests/slurm-tests/slurm_test_openmathreasoning && "
        f"python check_sft_results.py --workspace {args.workspace} "
    )

    run_cmd(
        ctx=wrap_arguments(checker),
        cluster=args.cluster,
        expname=f"check-sft-results-for-{args.backend}",
        log_dir=f"{args.workspace}/logs",
        run_after=[
            f"{args.expname_prefix}-baseline-summarize-results",
            f"{args.expname_prefix}-final-eval-summarize-results",
        ],
    )


if __name__ == "__main__":
    main()
