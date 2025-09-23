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

import logging
from dataclasses import dataclass
from typing import List, Optional

import typer

from nemo_skills.pipeline.app import typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_if_mounted,
    check_mounts,
    get_cluster_config,
    get_env_variables,
    get_exp,
    get_mounted_path,
    resolve_mount_paths,
    run_exp,
    temporary_env_update,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# Create a separate app for Megatron-LM
megatron_lm_app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@dataclass
class MegatronLMTask:
    """Configuration for Megatron-LM SFT training task."""

    # Only essential parameters that need validation/processing
    entrypoint: str
    pretrained_checkpoint: str
    tokenizer_model: str
    training_data: str
    validation_data: Optional[str]
    output_dir: str
    expname: str
    wandb_project: str
    wandb_group: Optional[str]
    disable_wandb: bool
    extra_arguments: str = ""

    def get_training_cmd(self) -> str:
        """Generate the Megatron-LM training command."""

        # Base command setup
        cmd = (
            "export UB_TIMEOUT=720 && "
            "export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
            "export NVTE_FWD_LAYERNORM_SM_MARGIN=16 && "
            "export NVTE_BWD_LAYERNORM_SM_MARGIN=16 && "
            "export NCCL_P2P_NET_CHUNKSIZE=2097152 && "
            "export NCCL_DEBUG=WARN && "
            "export TORCHINDUCTOR_WORKER_START=fork && "
            "echo 'Starting Megatron-LM SFT training' && "
            f"python -u {self.entrypoint} "
            f"    --sft "
            f"    --pretrained-checkpoint {self.pretrained_checkpoint} "
            f"    --tokenizer-model {self.tokenizer_model} "
            f"    --per-split-data-args-path {self.training_data} "
            f"    --load {self.output_dir}/checkpoints "
            f"    --save {self.output_dir}/checkpoints "
            f"    --tensorboard-dir {self.output_dir}/tensorboard "
            f"    --tensor-model-parallel-size {self.num_gpus} "
            f"    --data-cache-path {self.output_dir}/data_cache "
        )

        # Add wandb configuration if enabled
        if not self.disable_wandb:
            essential_args.extend([f"--wandb-project {self.wandb_project}", f"--wandb-exp-name {self.expname}"])
            if self.wandb_group:
                essential_args.append(f"--wandb-group {self.wandb_group}")

        # Add extra arguments if provided
        if self.extra_arguments:
            essential_args.extend(self.extra_arguments.split())

        # Format arguments with line continuations
        for i, arg in enumerate(essential_args):
            if i == len(essential_args) - 1:
                cmd_parts.append(f"    {arg}")
            else:
                cmd_parts.append(f"    {arg} \\")

        return "\n".join(cmd_parts)


def get_training_cmd(
    cluster_config,
    partition,
    pretrained_checkpoint,
    tokenizer_model,
    training_data,
    validation_data,
    num_gpus,
    num_nodes,
    expname,
    output_dir,
    disable_wandb,
    wandb_project,
    wandb_group,
    extra_arguments,
    log_dir,
    env_variables,
):
    """Generate Megatron-LM training command."""

    # timeout = get_timeout(cluster_config, partition)

    task = MegatronLMTask(
        entrypoint=entrypoint,
        pretrained_checkpoint=pretrained_checkpoint,
        tokenizer_model=tokenizer_model,
        training_data=training_data,
        validation_data=validation_data,
        output_dir=output_dir,
        expname=expname,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        disable_wandb=disable_wandb,
        extra_arguments=extra_arguments,
    )

    return task.get_training_cmd()


@megatron_lm_app.command(name="sft", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def sft_megatron_lm(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("megatron-lm-sft", help="Experiment name"),
    entrypoint: str = typer.Option("pretrain_mamba.py", help="Entrypoint script name"),
    pretrained_checkpoint: str = typer.Option(..., help="Path to the pretrained checkpoint"),
    tokenizer_model: str = typer.Option(..., help="Path to the tokenizer model"),
    training_data: str = typer.Option(..., help="Path to the training data"),
    validation_data: Optional[str] = typer.Option(None, help="Path to the validation data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    wandb_group: str = typer.Option(None, help="Weights & Biases group name."),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    skip_hf_home_check: bool = typer.Option(
        False,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Runs Megatron-LM SFT training.

    All extra arguments are passed directly to the Megatron-LM training script.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting Megatron-LM SFT training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if log_dir is None:
        log_dir = output_dir

    output_dir, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    # Check if paths are mounted
    if pretrained_checkpoint.startswith("/"):
        check_if_mounted(cluster_config, pretrained_checkpoint)
    if tokenizer_model.startswith("/"):
        check_if_mounted(cluster_config, tokenizer_model)
    if training_data.startswith("/"):
        training_data = get_mounted_path(cluster_config, training_data)
    if validation_data is not None and validation_data.startswith("/"):
        validation_data = get_mounted_path(cluster_config, validation_data)

    env_variables = get_env_variables(cluster_config)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        entrypoint=entrypoint,
        pretrained_checkpoint=pretrained_checkpoint,
        tokenizer_model=tokenizer_model,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        output_dir=output_dir,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        extra_arguments=extra_arguments,
        log_dir=f"{log_dir}/training-logs",
        env_variables=env_variables,
    )

    server_config = None
    env_update = {}

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        with temporary_env_update(cluster_config, env_update):
            for job_id in range(num_training_jobs):
                prev_task = add_task(
                    exp,
                    cmd=train_cmd,
                    task_name=f"{expname}-sft-{job_id}",
                    log_dir=f"{log_dir}/training-logs",
                    container=cluster_config["containers"]["megatron-lm"],
                    num_gpus=num_gpus,
                    num_nodes=num_nodes,
                    cluster_config=cluster_config,
                    server_config=server_config,
                    partition=partition,
                    time_min=time_min,
                    run_after=run_after,
                    reuse_code=reuse_code,
                    reuse_code_exp=reuse_code_exp,
                    task_dependencies=[prev_task] if prev_task is not None else None,
                    slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                    heterogeneous=True if server_config is not None else False,
                    with_sandbox=False,
                    with_ray=False,  # Megatron-LM doesn't use Ray
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                )
        run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    megatron_lm_app()
