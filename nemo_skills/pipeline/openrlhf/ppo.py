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
from typing import List

import os
import math
import nemo_run as run
import typer
from datetime import datetime
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import Optional

from nemo_skills.pipeline.openrlhf import openrlhf_app
from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.generate import wrap_cmd
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)

def convert_args_to_dict(args):
    """
    Converts a list of command-line arguments into a dictionary.
    
    Options that start with '--' or '-' are treated as keys.
    If the next token exists and is not another option, it is used as the value.
    If no value is provided, the key is set to True.
    Non-option arguments are collected under the key 'positional'.
    
    Args:
        args (list of str): The list of extra arguments, e.g.
                             ["arg1", "--foo", "bar", "arg2"]
    
    Returns:
        dict: A dictionary with the parsed arguments.
    """
    arg_dict = {}
    positional = []
    i = 0

    while i < len(args):
        arg = args[i]

        # Handle long options (e.g., --foo or --foo=bar)
        if arg.startswith("--"):
            # Remove the leading '--'
            key = arg[2:]
            if "=" in key:
                # Option provided as --key=value
                key, value = key.split("=", 1)
                arg_dict[key] = value
                i += 1
            else:
                # Check if the next token exists and is not an option
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    arg_dict[key] = args[i + 1]
                    i += 2
                else:
                    # No value provided; treat it as a boolean flag
                    arg_dict[key] = "True"
                    i += 1

        # Handle short options (e.g., -f or -f value)
        elif arg.startswith("-") and len(arg) > 1:
            # Remove the leading '-'
            key = arg[1:]
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                arg_dict[key] = args[i + 1]
                i += 2
            else:
                arg_dict[key] = True
                i += 1

        else:
            # Positional argument
            positional.append(arg)
            i += 1

    arg_dict["positional"] = positional

    return arg_dict

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


    def get_ray_server_cmd(self, start_cmd, cluster_config):
        ray_start_cmd = (
            "if [ \"${SLURM_PROCID:-0}\" = 0 ]; then "
            "    echo 'Starting head node' && "
            "    export RAY_raylet_start_wait_time_s=120 && "
            "    ray start "
            "        --head "
            "        --port=6379 "
            f"       {self.get_ray_server_ports(cluster_config)} && "
            f"   {start_cmd} ;"
            "else "
            "    echo 'Starting worker node' && "
            "    export RAY_raylet_start_wait_time_s=120 && "
            "    echo \"Connecting to head node at $SLURM_MASTER_NODE\" && "
            "    ray start "
            "        --block "
            "        --address=$SLURM_MASTER_NODE:6379 "
            f"       {self.get_ray_server_ports(cluster_config)} ;"
            "fi"
        )
        return ray_start_cmd

    def get_ray_server_ports(self, cluster_config):
        ports = (
            "--node-manager-port=12345 "
            "--object-manager-port=12346 "
            "--dashboard-port=8265 "
            "--dashboard-agent-grpc-port=12347 "
            "--runtime-env-agent-port=12349 "
            "--metrics-export-port=12350 "
            "--min-worker-port=14349 "
            "--max-worker-port=18349 "
        )
        return ports

    def get_ray_launch_cmd(self, cluster_config):
        cmd = (
            "ray job submit --address='http://127.0.0.1:8265' -- "
        )
        return cmd


    def get_default_reward_critic_args(self):
        cmd = {
            "reward_pretrain" : f"{self.reward_model}",
            "ref_num_nodes" : f"{self.num_nodes}",
            "ref_num_gpus_per_node" : f"{self.num_gpus}",
            "reward_num_nodes" : f"{self.num_nodes}",
            "reward_num_gpus_per_node" : f"{self.num_gpus}",
            "critic_num_nodes" : f"{self.num_nodes}",
            "critic_num_gpus_per_node" : f"{self.num_gpus}",
            "vllm_num_engines" : f"{self.num_gpus}",
            "vllm_tensor_parallel_size" : f"2",
            "colocate_critic_reward " : f"False",
            "colocate_actor_ref " : f"False",
        }
        return cmd

    def get_default_actor_args(self):
        cmd = {
            "actor_num_nodes" : f"{self.num_nodes}",
            "actor_num_gpus_per_node" : f"{self.num_gpus}",
        }
        return cmd


    def get_default_train_args(self):
        # NOTE:
        # `ckpt` refers to deepspeed intermediate checkpoints (the equivalent of nemo checkpoints saved during training,
        # with optim states)
        # `save` refers to the final HF model checkpoint (the equivalent of nemo final model checkpoint)
        # You can opt in to save both ds and HF checkpoint at every save_steps by setting `--save_hf_ckpt` as extra args
        cmd = {
            "pretrain" : f"{self.model}",
            "load_checkpoint" : None,
            "ckpt_path" : os.path.join(self.output_dir, 'ds_checkpoints'),
            "max_ckpt_num" : f"3",
            "max_ckpt_mem" : f"10000000000",
            "save_path" : os.path.join(self.output_dir, 'checkpoints'),
            "save_steps" : f"-1",
            "max_samples" : f"100000",
            "max_epochs" : f"1",
        }
        return cmd


    def get_default_data_args(self):
        # Note: Validation data isnt used as of now
        # If using chat message dict as data, add `--apply_chat_template`
        # and --input_key 'context_messages'
        cmd = {
            "prompt_data" : f"{self.prompt_data}",
            "input_key" : f"'question'",
            "input_template" : None,
        }

        return cmd

    def get_default_common_arg_overrides(self):
        cmd = {
            "actor_learning_rate" : f"5e-7",
            "critic_learning_rate" : f"9e-6",
            "train_batch_size" : f"128",
            "micro_train_batch_size" : f"8",
            "prompt_max_len" : f"1024",
            "generate_max_len" : f"1024",
            "logging_steps" : f"1",
            "eval_steps" : f"-1",
            "zero_stage" : f"3",
            "packing_samples" : f"False",
            "bf16" : f"False",
            "flash_attn" : f"False",
            "gradient_checkpointing" : f"False",
            "adam_offload" : f"False",
        }
        return cmd

    def get_default_common_rl_arg_overrides(self):
        cmd = {
            "micro_rollout_batch_size" : f"16",
            "rollout_batch_size" : f"1024",
            "n_samples_per_prompt" : f"1",
            "actor_learning_rate" : f"5e-7",
            "critic_learning_rate" : f"9e-6",
            "init_kl_coef" : f"0.01",
            "normalize_reward" : "False",
            "vllm_sync_backend" : f"nccl",
        }
        return cmd

    def format_wandb_args(self, disable_wandb, wandb_project, expname):
        if not disable_wandb:
            if os.getenv('WANDB_API_KEY') is None:
                raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")

            cmd = (f" --use_wandb $WANDB_API_KEY "
                   f" --wandb_project {wandb_project} "
                   f" --wandb_run_name {expname} ")
        else:
            cmd = ""

        return cmd

    def get_preamble_cmd(self, cluster_config):
        cmd = " echo 'No preamble command to execute, skipping...' "
        return cmd

    def format_args(self, extra_args):
        assert type(extra_args) == dict, "extra_args must be a dictionary"
        args = {}
        args.update(self.get_default_reward_critic_args())
        args.update(self.get_default_actor_args())
        args.update(self.get_default_train_args())
        args.update(self.get_default_data_args())
        args.update(self.get_default_common_arg_overrides())
        args.update(self.get_default_common_rl_arg_overrides())

        args.update(extra_args)

        flattened_args = args['positional'] + [f'--{k} {v}' if v != "True" else f'--{k}' for k, v in args.items() if (k != 'positional') and (v != "False") and (v != None)]
        args = f'{" ".join(flattened_args)}'
        return args


    def get_job_cmd(self, cluster_config):
        ray_job_cmd = self.get_ray_launch_cmd(cluster_config)
        formatted_args = self.format_args(self.extra_arguments)
        ray_job_cmd = (
            f"echo 'Starting training' && "
            f"{ray_job_cmd} python3 -m openrlhf.cli.train_ppo_ray "
            f"  {formatted_args} "
            f"  {self.logging_params} "
        )
        return ray_job_cmd

    def get_cmd(self, cluster_config):

        self.logging_params = self.format_wandb_args(self.disable_wandb, self.wandb_project, self.expname)
        preamble_cmd = self.get_preamble_cmd(cluster_config)

        cmd = (
            f"export HYDRA_FULL_ERROR=1 && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
            f"export TRITON_CACHE_DIR=/nemo_run/code/.triton_cache && "
            f"cd /nemo_run/code && "
            f"echo 'Running preamble command:' {preamble_cmd} && "
            f"{preamble_cmd} && "
        )

        ray_job_cmd = self.get_job_cmd(cluster_config)
        ray_server_cmd = self.get_ray_server_cmd(ray_job_cmd, cluster_config)

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

    cmd = task.get_cmd(cluster_config)
    print(cmd)
    return cmd


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
    final_checkpoint_path: str = typer.Option(None, help="Where to put the final checkpoint"),
    expname: str = typer.Option(..., help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    rm_model: str = typer.Option(..., help="Path to the HF reward model"),
    prompt_data: str = typer.Option(None, help="Path to the prompt data"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    with_sandbox: bool = typer.Option(False, help="If sandbox is required for code generation"),
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
    extra_arguments = convert_args_to_dict(ctx.args)
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    num_nodes = {k.split('_')[0] : int(v) for k, v in extra_arguments.items() if k.endswith('_num_nodes')}
    num_gpus = {k.split('_')[0] : int(v) for k, v in extra_arguments.items() if k.endswith('_num_gpus_per_node')}

    for k in ['ref', 'reward', 'critic', 'actor']:
        assert k in num_nodes, f"Missing {k} in num_nodes, need num_nodes and num_gpus for ref, reward, critic, actor"
        assert k in num_gpus, f"Missing {k} in num_gpus, need num_nodes and num_gpus for ref, reward, critic, actor"

    num_gpus = sum((num_gpus[k] * num_nodes[k] for k in num_gpus))
    num_nodes = math.ceil(num_gpus / 8)
    num_gpus = num_gpus if num_nodes == 1 else 8

    print("Total number of GPUs per node: ", num_gpus)
    print("Total number of nodes: ", num_nodes)

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
        if prompt_data.startswith('/'):
            check_if_mounted(cluster_config, prompt_data)

    # if not final_checkpoint_path:
    #     final_checkpoint_path = f"{output_dir}/model-averaged-hf"
    # check_if_mounted(cluster_config, final_checkpoint_path)

    if cluster_config["executor"] == "local":
        assert "HF_HOME" in os.environ, "HF_HOME must be set when running locally"

    # if " " in str(average_steps):
    #     raise ValueError("average steps should be separated with commas")

    # Check if custom PPOOpenRLHFTask is provided via ctx.obj['ppo_task'], use that if available
    if hasattr(ctx, 'obj') and ctx.obj is not None and isinstance(ctx.obj, dict) and 'ppo_task' in ctx.obj:
        ppo_task = ctx.obj['ppo_task']  # type: PPOOpenRLHFTask
        assert isinstance(ppo_task,
                          PPOOpenRLHFTask), "`ppo_task` must be a subclass of PPOOpenRLHFTask"
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
                num_tasks=1,
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                with_sandbox=with_sandbox,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )

        # cmd = get_avg_checkpoints_cmd(
        #     nemo_model=nemo_model,
        #     output_dir=output_dir,
        #     final_nemo_path=final_nemo_path,
        #     average_steps=f"--steps {' '.join(average_steps.split(','))} " if average_steps else "",
        # )

        # add_task(
        #     exp,
        #     cmd=cmd,
        #     task_name=f"{expname}-prepare-eval",
        #     log_dir=f"{log_dir}/prepare-eval-logs",
        #     container=cluster_config["containers"]['nemo'],
        #     cluster_config=cluster_config,
        #     partition=partition,
        #     time_min=time_min,
        #     num_nodes=1,
        #     num_tasks=1,
        #     num_gpus=num_gpus,
        #     run_after=run_after,
        #     reuse_code=reuse_code,
        #     reuse_code_exp=reuse_code_exp,
        #     task_dependencies=[prev_task] if prev_task is not None else None,
        #     slurm_kwargs={"exclusive": exclusive} if exclusive else None,
        # )
        #
        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()


