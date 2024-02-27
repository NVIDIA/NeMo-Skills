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

import sys
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, launch_job

from nemo_skills.utils import setup_logging

SLURM_CMD = (
    "python /code/nemo_skills/finetuning/average_checkpoints.py "
    "    --untarred_nemo_folder /nemo_model "
    "    --name_prefix=model "
    "    --checkpoint_dir=/training_folder "
    "{conversion_step}"
)
MOUNTS = (
    "{NEMO_SKILLS_CODE}:/code,"
    "{training_folder}:/training_folder,"
    "{output_path}:/inference_folder,"
    "{nemo_model}:/nemo_model"
)
JOB_NAME = "prepare-for-eval-{inference_model_name}"


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--training_folder", required=True)
    parser.add_argument("--server_type", choices=('nemo',), default='nemo')
    parser.add_argument("--output_path", required=True, help="Path to save the TensorRT-LLM model")
    parser.add_argument("--nemo_model", required=True, help="Only need this to get the config file")
    parser.add_argument("--num_gpus", required=True, type=int)
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    args = parser.parse_args()

    args.training_folder = Path(args.training_folder).absolute()
    args.output_path = Path(args.output_path).absolute()
    model_name = args.output_path.name

    if model_name.endswith('.nemo'):
        # mounting the parent folder instead
        args.output_path = args.output_path.parent
    args.output_path.mkdir(exist_ok=True, parents=True)

    conversion_step = f"&& mv /training_folder/model-averaged.nemo /inference_folder/{model_name}"
    # TODO: add NeMo to TensorRT-LLM conversion step

    format_dict = {
        "training_folder": args.training_folder,
        "inference_model_name": model_name,
        "nemo_model": args.nemo_model,
        "output_path": args.output_path,
        "conversion_step": conversion_step,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
    }

    launch_job(
        cmd=SLURM_CMD.format(**format_dict),
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=args.num_gpus,
        job_name=JOB_NAME.format(**format_dict),
        container=CLUSTER_CONFIG["containers"][args.server_type],
        mounts=MOUNTS.format(**format_dict),
        partition=args.partition,
        with_sandbox=False,
    )
