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
import os
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def prepare_questions(cluster, expname, output_dir: str):
    """
    Prepare questions for the Open Code Reasoning task.

    Args:
        output_dir (str): Directory to save the prepared questions.
    """
    command = ("python /nemo_run/code/recipes/opencodereasoning/scripts/prepare_questions.py "
               f"--output_dir '{output_dir}'")

    run_cmd(
        ctx=wrap_arguments(command),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=None,
        check_mounted_paths=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Open Code Reasoning questions")

    parser.add_argument('--cluster', type=str, required=True,
                        help="Cluster name to run the job on.")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Mounted Directory to save the prepared questions.",
    )

    args = parser.parse_args()

    if not 'HF_HOME' in os.environ:
        print("HF_HOME environment variable not set in the enviroment, dataset cache will NOT be used.")

    prepare_questions(
        cluster=args.cluster,
        expname="prepare_questions",
        output_dir=args.output_dir,
    )
