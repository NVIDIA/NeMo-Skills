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
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.openrlhf import openrlhf_app
from nemo_skills.pipeline.utils import get_ray_server_cmd
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


@dataclass
class PPOOpenRLHFTask:
    model: str
    reward_model: str
    output_dir: str
    prompt_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    disable_wandb: bool
    wandb_project: str
    timeout: str
    extra_arguments: str = ""
    logging_params: str = ""

    def get_ray_launch_cmd(self):
        cmd = "ray job submit --address='http://127.0.0.1:8265' -- "
        return cmd

    def format_reward_critic_args(self):
        cmd = (
            f" --reward_pretrain {self.reward_model} "
            # TODO: add proper defaults when we figure out how these should be used
            #       for now we require users to be explicit
        )
        return cmd

    def format_actor_args(self):
        # TODO: add proper defaults when we figure out how these should be used
        #       for now we require users to be explicit
        # cmd = f" --actor_num_nodes {self.num_nodes} --actor_num_gpus_per_node {self.num_gpus} "
        cmd = ""
        return cmd

    def format_train_args(self):
        # NOTE:
        # `ckpt` refers to deepspeed intermediate checkpoints (the equivalent of nemo checkpoints saved during training,
        # with optim states)
        # `save` refers to the final HF model checkpoint (the equivalent of nemo final model checkpoint)
        # You can opt in to save both ds and HF checkpoint at every save_steps by setting `--save_hf_ckpt` as extra args
        cmd = (
            f" --pretrain {self.model} "
            f" --load_checkpoint "
            f" --ckpt_path {os.path.join(self.output_dir, 'ds_checkpoints')} "
            f" --max_ckpt_num 3 "
            f" --max_ckpt_mem 10000000000 "
            f" --save_path {os.path.join(self.output_dir, 'checkpoints')} "
            f" --save_steps -1 "
            f" --max_samples 500000 "
            f" --max_epochs 1 "
        )
        return cmd

    def format_data_args(self):
        # Note: Validation data isnt used as of now
        # If using chat message dict as data, add `--apply_chat_template`
        # and --input_key 'context_messages'
        cmd = f" --prompt_data {self.prompt_data} --input_key 'context_messages' " 
        return cmd

    def get_common_arg_overrides(self):
        cmd = (
            " --train_batch_size 128 "
            " --micro_train_batch_size 8 "
            " --prompt_max_len 1024 "
            " --generate_max_len 1024 "
            " --logging_steps 1 "
            " --eval_steps -1 "
            " --zero_stage 3 "
            " --packing_samples "
            " --bf16 "
            " --flash_attn "
            " --gradient_checkpointing "
            " --adam_offload "
        )
        return cmd

    def get_common_rl_arg_overrides(self):
        cmd = (
            " --micro_rollout_batch_size 16 "
            " --rollout_batch_size 1024 "
            " --n_samples_per_prompt 1 "
            " --actor_learning_rate 5e-7 "
            " --critic_learning_rate 9e-6 "
            " --init_kl_coef 0.01 "
            " --normalize_reward "
            " --vllm_sync_backend nccl "
        )
        return cmd

    def format_wandb_args(self, disable_wandb, wandb_project, expname):
        if not disable_wandb:
            if os.getenv('WANDB_API_KEY') is None:
                raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")

            cmd = f" --use_wandb $WANDB_API_KEY " f" --wandb_project {wandb_project} " f" --wandb_run_name {expname} "
        else:
            cmd = ""

        return cmd

    def get_preamble_cmd(self):
        cmd = " echo 'No preamble command to execute, skipping...' "
        return cmd

    def get_job_cmd(self):
        ray_job_cmd = self.get_ray_launch_cmd()
        ray_job_cmd = (
            f"echo 'Starting training' && "
            f"{ray_job_cmd} python3 -m openrlhf.cli.train_ppo_ray "
            f"  {self.format_reward_critic_args()} "
            f"  {self.format_actor_args()} "
            f"  {self.format_train_args()} "
            f"  {self.format_data_args()} "
            f"  {self.get_common_arg_overrides()} "
            f"  {self.get_common_rl_arg_overrides()} "
            f"  {self.logging_params} "
            f"  {self.extra_arguments} "
        )
        return ray_job_cmd

    def get_cmd(self):

        self.logging_params = self.format_wandb_args(self.disable_wandb, self.wandb_project, self.expname)
        preamble_cmd = self.get_preamble_cmd()

        cmd = (
            f"export HYDRA_FULL_ERROR=1 && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
            f"cd /nemo_run/code && "
            f"{preamble_cmd} && "
        )

        ray_job_cmd = self.get_job_cmd()
        ray_server_cmd = get_ray_server_cmd(ray_job_cmd)

        cmd = f"{cmd} {ray_server_cmd} "
        return cmd


def get_training_cmd(
    cluster_config,
    task: Optional[PPOOpenRLHFTask],
    partition,
    hf_model,
    rm_model,
    output_dir,
    prompt_data,
    num_gpus,
    num_nodes,
    expname,
    disable_wandb,
    wandb_project,
    extra_arguments,
):
    # TODO: use those
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition or cluster_config["partition"]]

        # subtracting 15 minutes to account for the time it takes to save the model
        # the format expected by nemo is days:hours:minutes:seconds
        time_diff = datetime.strptime(timeout, "%H:%M:%S") - datetime.strptime("00:15:00", "%H:%M:%S")
        timeout = (
            f'00:{time_diff.seconds // 3600:02d}:{(time_diff.seconds % 3600) // 60:02d}:{time_diff.seconds % 60:02d}'
        )

    if task is None:
        task = PPOOpenRLHFTask(
            model=hf_model,
            reward_model=rm_model,
            output_dir=output_dir,
            prompt_data=prompt_data,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            expname=expname,
            disable_wandb=disable_wandb,
            wandb_project=wandb_project,
            timeout=timeout,
            extra_arguments=extra_arguments,
            logging_params="",  # Updated later
        )

    else:
        task.timeout = timeout
        task.extra_arguments = extra_arguments

    return task.get_cmd()


def pack_nodes(gpus_per_node, num_nodes):
    order = ['actor', 'ref', 'critic', 'reward', 'vllm'] # This is the placement order in OpenRLHF cli/train_ppo_ray.py and so we follow that
    items = []
    for model in order:
        g = gpus_per_node[model]
        n = num_nodes[model]
        if g*n == 0:
            continue
        items.extend([g]*n)

    bins = []  # Each element is the sum of GPUs allocated on that node.
    for item in items:
        placed = False
        # Try to place the item in an existing node
        for i in range(len(bins)):
            if bins[i] + item <= 8:
                bins[i] += item
                placed = True
                break
        # If it doesn't fit in any existing node, allocate a new node.
        if not placed:
            bins.append(item)

    num_nodes_needed = len(bins)
    return num_nodes_needed


def get_num_nodes_and_gpus(
    cluster_config,
    rm_model,
    num_processes,
    gpus_per_node,
    colocate_critic_reward,
    colocate_actor_ref,
    advantage_estimator,
):
    if advantage_estimator not in ['gae']:
        # Colocation does not make sense if critic model is not needed
        # This modification is also necessary for the next reward allocation if condition
        gpus_per_node['critic'] = 0
        num_processes['critic'] = 0
        colocate_critic_reward = False

    try:
        check_if_mounted(cluster_config, rm_model)
        assert not colocate_critic_reward
    except:
        gpus_per_node['reward'] = 0
        num_processes['reward'] = 0

    if colocate_actor_ref:
        gpus_per_node['ref'] = 0
        num_processes['ref'] = 0

    num_nodes = pack_nodes(gpus_per_node, num_processes)
    num_gpus = 8

    return num_nodes, num_gpus


@openrlhf_app.command(name='ppo', context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def ppo_openrlhf(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("openrlhf-ppo", help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    rm_model: str = typer.Option(..., help="Path to the HF reward model"),
    prompt_data: str = typer.Option(None, help="Path to the prompt data"),
    ref_num_nodes: int = typer.Option(1, help="Number of nodes for reference model"),
    ref_num_gpus_per_node: int = typer.Option(..., help="Number of GPUs per node for reference model"),
    actor_num_nodes: int = typer.Option(..., help="Number of nodes for actor model"),
    actor_num_gpus_per_node: int = typer.Option(..., help="Number of GPUs per node for actor model"),
    critic_num_nodes: int = typer.Option(..., help="Number of nodes for critic model"),
    critic_num_gpus_per_node: int = typer.Option(..., help="Number of GPUs per node for critic model"),
    reward_num_nodes: int = typer.Option(..., help="Number of nodes for reward model"),
    reward_num_gpus_per_node: int = typer.Option(..., help="Number of GPUs per node for reward model"),
    vllm_num_engines: int = typer.Option(..., help="Number of VLLM engines"),
    vllm_tensor_parallel_size: int = typer.Option(..., help="Number of VLLM tensor parallel size"),
    colocate_critic_reward: bool = typer.Option(False, help="Colocate critic and reward models"),
    colocate_actor_ref: bool = typer.Option(False, help="Colocate actor and reference models"),
    advantage_estimator: str = typer.Option('gae', help="Path to the advantage estimator model"),
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
    """Run a pre-defined module or script in the NeMo-Skills container."""
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    assert len(rm_model.split(',')) == 1, f"RM model must be a single model as behavior as our team has not tested multi-model RM feature in OpenRLHF. We got RM models: {rm_model}"

    cluster_config = get_cluster_config(cluster, config_dir)

    gpus_per_node = {
        'actor': actor_num_gpus_per_node,
        'critic': critic_num_gpus_per_node,
        'reward': reward_num_gpus_per_node,
        'ref': ref_num_gpus_per_node,
        'vllm': vllm_tensor_parallel_size,
    }

    num_processes = {
        'actor': actor_num_nodes,
        'critic': critic_num_nodes,
        'reward': reward_num_nodes,
        'ref': ref_num_nodes,
        'vllm': vllm_num_engines,
    }

    for model in gpus_per_node:
        if model == 'vllm':
            continue
        extra_arguments += f" --{model}_num_nodes {num_processes[model]}"
        extra_arguments += f" --{model}_num_gpus_per_node {gpus_per_node[model]}"
    extra_arguments += f" --vllm_num_engines {vllm_num_engines}"
    extra_arguments += f" --vllm_tensor_parallel_size {vllm_tensor_parallel_size}"

    num_nodes, num_gpus = get_num_nodes_and_gpus(
        cluster_config,
        rm_model,
        num_processes,
        gpus_per_node,
        colocate_critic_reward,
        colocate_actor_ref,
        advantage_estimator,
    )
    LOG.info(f"Total number of nodes: {num_nodes}")
    LOG.info(f"Number of GPUs per node: {num_gpus}")
    extra_arguments += f" --advantage_estimator {advantage_estimator}"
    if colocate_actor_ref:
        LOG.warning("Colocating actor and reference models on same GPUs.")
        extra_arguments += f" --colocate_actor_ref"
    if colocate_critic_reward:
        LOG.warning("Colocating critic and reward models on same GPUs")
        extra_arguments += f" --colocate_critic_reward"

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

    if cluster_config["executor"] == "local":
        assert "HF_HOME" in os.environ, "HF_HOME must be set when running locally"

    # Check if custom PPOOpenRLHFTask is provided via ctx.obj['ppo_task'], use that if available
    if hasattr(ctx, 'obj') and ctx.obj is not None and isinstance(ctx.obj, dict) and 'ppo_task' in ctx.obj:
        ppo_task = ctx.obj['ppo_task']  # type: PPOOpenRLHFTask
        assert isinstance(ppo_task, PPOOpenRLHFTask), "`ppo_task` must be a subclass of PPOOpenRLHFTask"
    else:
        ppo_task = None

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        task=ppo_task,
        partition=partition,
        hf_model=hf_model,
        rm_model=rm_model,
        output_dir=output_dir,
        prompt_data=prompt_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        extra_arguments=extra_arguments,
    )

    print(train_cmd)

    with run.Experiment(expname) as exp:
        prev_task = None
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{expname}-ppo-{job_id}',
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["vllm"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=1,  # torchrun will launch all processes
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )

        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
