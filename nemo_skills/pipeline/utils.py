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

import logging
import os
from pathlib import Path

import nemo_run as run
from huggingface_hub import get_token

# from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.slurm import JobPaths

LOG = logging.getLogger(__file__)


def get_generation_command(server_address, generation_commands):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        # might be required if we are not hosting server ourselves
        f"export NVIDIA_API_KEY={os.getenv('NVIDIA_API_KEY', '')} && "
        f"export OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')} && "
        # this will try to handshake in a loop and unblock when the server responds
        f"echo 'Waiting for the server to start' && "
        f"while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done && "
        # will run in a single task always (no need to check mpi env vars)
        f"{generation_commands}"
    )
    return cmd


def get_server_command(server_type: str, num_gpus: int, num_nodes: int, model_path: str, cluster_config: dict):
    num_tasks = num_gpus
    if server_type == 'nemo':
        server_start_cmd = (
            f"python /nemo_run/code/nemo_skills/inference/server/serve_nemo.py gpt_model_file={model_path} "
            f"trainer.devices={num_gpus} "
            f"trainer.num_nodes={num_nodes} "
            f"tensor_model_parallel_size={num_gpus} "
            f"pipeline_model_parallel_size={num_nodes} "
        )
        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] == "local":
            num_tasks = 1

    elif server_type == 'vllm':
        server_start_cmd = (
            f"NUM_GPUS={num_gpus} bash /nemo_run/code/nemo_skills/inference/server/serve_vllm.sh "
            f"{model_path} {os.path.basename(model_path)} 0 openai 5000"
        )

        if os.environ.get("MAX_SEQ_LEN", None) is not None:
            server_start_cmd = f"export MAX_SEQ_LEN={os.environ['MAX_SEQ_LEN']} && {server_start_cmd}"

        num_tasks = 1
    else:
        # adding sleep to ensure the logs file exists
        # need this flag for stable Nemotron-4-340B deployment
        server_start_cmd = (
            f"FORCE_NCCL_ALL_REDUCE_STRATEGY=1 python /nemo_run/code/nemo_skills/inference/server/serve_trt.py "
            f"--model_path {model_path}"
        )
        num_tasks = num_gpus

    server_cmd = (
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export HF_TOKEN={get_token()} && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_sandox_command():
    return "/entrypoint.sh && /start.sh"


# def get_logs_cls(cluster_config, expname):
#     class MainJobPaths(JobPaths):
#         @property
#         def stdout(self) -> Path:
#             return Path(f"{cluster_config['workspace']}/{expname}" / "slurm-logs" / "sbatch.txt")

#         @property
#         def srun_stdout(self) -> Path:
#             return Path(f"{cluster_config['workspace']}/{expname}" / "slurm-logs" / "job_logs.txt")

#     return MainJobPaths


def get_executor(
    cluster_config,
    expname,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    partition=None,
):
    mounts = cluster_config.get('mounts', []) + [f"{cluster_config['workspace']}/{expname}:/exp"]
    if cluster_config["executor"] == "local":
        # creating a folder
        os.makedirs(f"{cluster_config['workspace']}/{expname}", exist_ok=True)
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")
        return DockerExecutor(
            container_image=container,
            packager=run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl'),
            ipc_mode="host",
            volumes=cluster_config.get('mounts', []),
            ntasks_per_node=tasks_per_node,
            num_gpus=gpus_per_node,
            env_vars={"PYTHONUNBUFFERED": "1"},  # this makes sure logs are streamed right away
        )

    # creating a folder - need to do it through Tunnel to ensure it's on the remote machine
    # TODO: reuse the tunnel
    tunnel = run.SSHTunnel(**cluster_config["ssh_tunnel"])
    tunnel.run(f"mkdir -p {cluster_config['workspace']}/{expname}")

    partition = partition or cluster_config.get("partition")
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=tunnel,
        container_image=container,
        container_mounts=mounts,
        time=timeout,
        packager=run.GitArchivePackager(include_pattern='nemo_skills/dataset/**/*.jsonl'),
        gpus_per_node=gpus_per_node,
        job_name_prefix=cluster_config["job_name_prefix"],
        srun_args=[
            "--no-container-mount-home",
            "--overlap",
            "--mpi=pmix",
            '--wait=10',
            # we need to be explicit about this in srun as commands might need to run in parallel
            f"--ntasks={tasks_per_node * num_nodes}",
            f"--nodes={num_nodes}",
        ],
        # TODO: can we relax this to allow partial node allocation?
        exclusive=True,
        mem=0,
        # job_paths_cls=get_logs_cls(cluster_config, expname),
        # job_paths_cls=MainJobPaths,
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
    )


def add_task(
    exp,
    cmd,
    task_name,
    cluster_config,
    container,
    # TODO: are these good defaults?
    num_tasks=1,
    num_gpus=1,
    num_nodes=1,
    partition=None,
    with_sandbox=False,
    server_config=None,
):
    commands = []
    executors = []
    # assuming server always has the largest resources request, so it needs to go first
    if server_config is not None:
        server_cmd, num_server_tasks = get_server_command(**server_config, cluster_config=cluster_config)
        if 'container' not in server_config:
            server_container = cluster_config["containers"][server_config['server_type']]
        server_executor = get_executor(
            cluster_config=cluster_config,
            expname=exp._title,
            container=server_container,
            num_nodes=server_config['num_nodes'],
            tasks_per_node=num_server_tasks,
            gpus_per_node=server_config['num_gpus'],
            partition=partition,
        )
        commands.append(server_cmd)
        executors.append(server_executor)

    # then goes the main task unless it's empty
    if cmd:
        commands.append(cmd)
        executors.append(
            get_executor(
                cluster_config=cluster_config,
                expname=exp._title,
                container=container,
                num_nodes=num_nodes,
                tasks_per_node=num_tasks,
                gpus_per_node=num_gpus,
                partition=partition,
            )
        )

    # finally a sandbox if needed
    if with_sandbox:
        sandbox_executor = get_executor(
            cluster_config=cluster_config,
            expname=exp._title,
            container=cluster_config["containers"]["sandbox"],
            num_nodes=executors[0].nodes if cluster_config["executor"] == "slurm" else 1,
            tasks_per_node=1,
            gpus_per_node=num_gpus,
            partition=partition,
        )
        # annoyingly named differently in nemo.run
        if cluster_config["executor"] == "local":
            sandbox_executor.volumes = []
        else:
            sandbox_executor.mounts = []  # we don't want to mount anything
        commands.append(get_sandox_command())
        executors.append(sandbox_executor)

    commands = [cmd.replace("$", "\\$") for cmd in commands]
    exp.add(
        [run.Script(inline=command) for command in commands],
        executor=executors,
        name=task_name,
    )


def run_exp(exp, cluster_config):
    if cluster_config['executor'] == 'local':
        exp.run(detach=False, tail_logs=True)
    else:
        exp.run(detach=True)
