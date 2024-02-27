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
from datetime import datetime
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, fill_env_vars, launch_job

from nemo_skills.utils import setup_logging

# note that we are using custom config nemo_skills/finetuning/sft_config.py
# which contains most of the important parameters
SLURM_CMD = """
export WANDB_API_KEY={WANDB_API_KEY} \
&& export HYDRA_FULL_ERROR=1 \
&& echo "Starting training" \
&& export PYTHONPATH=$PYTHONPATH:/code \
&& NVTE_APPLY_QK_LAYER_SCALING=1 python /code/nemo_skills/finetuning/start_sft.py \
    --config-name={config_name} --config-path={config_path} \
    model.tensor_model_parallel_size={num_gpus} \
    trainer.devices={num_gpus} \
    trainer.num_nodes={num_nodes} \
    model.restore_from_path=/nemo_model \
    model.data.validation_ds.file_path=/code/datasets/{validation_dataset}/validation-sft.jsonl \
    {logging_params} \
    exp_manager.name={expname} \
    exp_manager.explicit_log_dir=/results \
    exp_manager.exp_dir=/results \
    ++exp_manager.max_time_per_run={timeout} \
    {extra_arguments}
"""
MOUNTS = "{NEMO_SKILLS_CODE}:/code,{checkpoints_folder}:/results,{NEMO_SKILLS_DATA}:/data,{nemo_model}:/nemo_model"
JOB_NAME = "sft-{expname}"


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--project", default="nemo-skills-exps")
    parser.add_argument("--expname", required=True, help="Experiment name for logging purposes")
    parser.add_argument("--checkpoints_folder", required=True)
    parser.add_argument("--nemo_model", required=True)
    # have to be handled explicitly since hydra requires these to be first arguments
    parser.add_argument("--config-name", "-cn", default='sft_config')
    parser.add_argument("--config-path", "-cp", default='/code/nemo_skills/finetuning/')
    parser.add_argument(
        "--validation_dataset",
        default="gsm8k",
        help="Validation dataset to use. Make sure it exists inside datasets folder",
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable wandb logging and use tensorboard instead"
    )
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    args.checkpoints_folder = Path(args.checkpoints_folder).absolute()
    args.nemo_model = Path(args.nemo_model).absolute()

    args.checkpoints_folder.mkdir(exist_ok=True, parents=True)

    if 'timeouts' not in CLUSTER_CONFIG:
        timeout = "10000:00:00:00"
    else:
        timeout = CLUSTER_CONFIG["timeouts"][args.partition or CLUSTER_CONFIG["partition"]]
        # subtracting 15 minutes to account for the time it takes to save the model
        # the format expected by nemo is days:hours:minutes:seconds
        timeout = f'00:{datetime.strptime(timeout, "%H:%M:%S") - datetime.strptime("00:15:00", "%H:%M:%S")}'

    format_dict = {
        "project": args.project,
        "expname": args.expname,
        "config_name": args.config_name,
        "config_path": args.config_path,
        "checkpoints_folder": args.checkpoints_folder,
        "validation_dataset": args.validation_dataset,
        "nemo_model": args.nemo_model,
        "num_nodes": args.num_nodes,
        "num_gpus": args.num_gpus,
        "extra_arguments": extra_arguments,
        "timeout": timeout,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
    }
    fill_env_vars(format_dict, ["NEMO_SKILLS_DATA"])
    if not args.disable_wandb:
        fill_env_vars(format_dict, ["WANDB_API_KEY"])
        logging_params = (
            "exp_manager.create_wandb_logger=True "
            "exp_manager.wandb_logger_kwargs.name={expname} "
            "exp_manager.wandb_logger_kwargs.project={project} "
            "+exp_manager.wandb_logger_kwargs.id={expname} "
            "+exp_manager.wandb_logger_kwargs.resume=True "
        ).format(**format_dict)
    else:
        format_dict["WANDB_API_KEY"] = "n/a"
        logging_params = "exp_manager.create_wandb_logger=False +exp_manager.create_tensorboard_logger=True"
    format_dict["logging_params"] = logging_params

    launch_job(
        cmd=SLURM_CMD.format(**format_dict),
        num_nodes=args.num_nodes,
        tasks_per_node=args.num_gpus if CLUSTER_CONFIG["cluster"] == "slurm" else 1,
        gpus_per_node=args.num_gpus,
        job_name=JOB_NAME.format(**format_dict),
        container=CLUSTER_CONFIG["containers"]["nemo"],
        mounts=MOUNTS.format(**format_dict),
        partition=args.partition,
        with_sandbox=True,
    )
