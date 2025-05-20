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

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_if_mounted,
    get_cluster_config,
    get_exp,
    get_free_port,
    get_timeout,
    run_exp,
)
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


@dataclass
class NemoRLTask:
    model: str
    output_dir: str
    prompt_data: str
    eval_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    disable_wandb: bool
    wandb_project: str
    timeout: str
    log_dir: str
    extra_arguments: str = ""
    tmpdir: str = "/tmp"

    def format_train_args(self):
        cmd = (
            f"cluster.gpus_per_node={self.num_gpus} "
            f"cluster.num_nodes={self.num_nodes} "
            f"logger.log_dir={self.log_dir} "
            f"checkpointing.checkpoint_dir={self.output_dir}/checkpoints "
        )
        return cmd

    def format_data_args(self):
        cmd = (
            # "data.prompt_file=/opt/nemo-rl/examples/prompts/cot.txt "
        )
        return cmd

    def format_wandb_args(self, disable_wandb, wandb_project, expname):
        cmd = (
            "logger.wandb_enabled=true " if not disable_wandb else ""
            f"logger.wandb.project={wandb_project} "
            f"logger.wandb.name={expname} "
        )
        return cmd

    def get_preamble_cmd(self):
        cmd = " echo 'No preamble command to execute, skipping...' "
        return cmd

    def get_script_module(self):
        return "/nemo_run/code/nemo_skills/training/nemo_rl/start_sft_nemo_rl.py "

    def get_job_cmd(self):
        ray_job_cmd = (
            f"echo 'Starting training' && "
            # f"ls /nemo_run/code && "
            f"uv run --active python {self.get_script_module()} "
            f"  {self.format_train_args()} "
            # f"  {self.format_data_args()} "
            f"  {self.logging_params} "
            f"  {self.extra_arguments} "
        )
        return ray_job_cmd

    def get_cmd(self):

        self.logging_params = self.format_wandb_args(self.disable_wandb, self.wandb_project, self.expname)
        preamble_cmd = self.get_preamble_cmd()

        cmd = (
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code:/opt/nemo-rl && "
            f"export NEMO_RL_VENV_DIR=/opt/nemo_rl_venv && "
            f"export UV_CACHE_DIR={self.tmpdir}/uv && "
            # f"cd /nemo_run/code && "
            f"cd /opt/nemo-rl && "
            f"{preamble_cmd} && "
        )

        ray_job_cmd = self.get_job_cmd()

        cmd = f"{cmd} {ray_job_cmd} "
        return cmd


def get_training_cmd(
    cluster_config,
    task: Optional[NemoRLTask],
    partition,
    hf_model,
    output_dir,
    prompt_data,
    eval_data,
    num_gpus,
    num_nodes,
    expname,
    disable_wandb,
    wandb_project,
    extra_arguments,
    log_dir,
):
    # TODO: use those
    timeout = get_timeout(cluster_config, partition)

    if task is None:
        task = NemoRLTask(
            model=hf_model,
            output_dir=output_dir,
            prompt_data=prompt_data,
            eval_data=eval_data,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            expname=expname,
            disable_wandb=disable_wandb,
            wandb_project=wandb_project,
            timeout=timeout,
            extra_arguments=extra_arguments,
            log_dir=log_dir,
        )

    else:
        task.timeout = timeout
        task.extra_arguments = extra_arguments

    return task.get_cmd()


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"
    sglang = "sglang"


@app.command(name='sft', context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def sft_nemo_rl(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("openrlhf-ppo", help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    prompt_data: str = typer.Option(None, help="Path to the prompt data"),
    eval_data: str = typer.Option(None, help="Path to the eval data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
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
):
    """Runs Verl PPO training (verl.trainer.main_ppo)"""
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    check_if_mounted(cluster_config, hf_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = output_dir

    if num_training_jobs > 0:
        if prompt_data is None:
            raise ValueError("prompt_data is required when num_training_jobs > 0")
        if prompt_data.startswith("/"):  # could ask to download from HF
            check_if_mounted(cluster_config, prompt_data)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        task=None,
        partition=partition,
        hf_model=hf_model,
        output_dir=output_dir,
        prompt_data=prompt_data,
        eval_data=eval_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        extra_arguments=extra_arguments,
        log_dir=f"{log_dir}/training-logs",
    )

    server_config = None
    with get_exp(expname, cluster_config) as exp:
        prev_task = None
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{expname}-sft-{job_id}',
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["nemo-rl"],
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
                with_ray=True,
            )
        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
