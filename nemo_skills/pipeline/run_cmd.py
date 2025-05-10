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
from typing import List, Optional

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import get_exp, run_exp, add_task
from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills import utils
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_cmd(command, chunk_id=None, num_chunks=None, preprocess_cmd=None, postprocess_cmd=None):

    cmd = (f"export HYDRA_FULL_ERROR=1 && "
           f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code "
           f"&& cd /nemo_run/code && ")

    if chunk_id is not None and num_chunks is not None:
        cmd += f"export CHUNK_ID={chunk_id} && export NUM_CHUNKS={num_chunks} && "

    if preprocess_cmd is not None:
        cmd += f"{preprocess_cmd.strip()} && "

    cmd += f"{command.strip()} "

    if postprocess_cmd is not None:
        cmd += f"&& {postprocess_cmd.strip()} "

    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def run_cmd(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    command: str = typer.Option(None, help="Command to run in the container"),
    container: str = typer.Option("nemo-skills", help="Container to use for the run"),
    expname: str = typer.Option("script", help="Nemo run experiment name"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    num_gpus: int | None = typer.Option(None, help="Number of GPUs to use"),
    num_nodes: int = typer.Option(1, help="Number of nodes to use"),
    model: str = typer.Option(None, help="Path to the model to evaluate"),
    server_address: str = typer.Option(None, help="Address of the server hosting the model"),
    server_type: str = typer.Option(None, help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    num_chunks: int = typer.Option(
        None,
        help="Number of chunks to compute. If None, will not chunk the dataset.",
    ),
    chunk_ids: str = typer.Option(
        None,
        help="List of explicit chunk ids to run. Separate with , or .. to specify range. "
             "Can provide a list directly when using through Python",
    ),
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
    preprocess_cmd: str = typer.Option(None, help="Command to run before job"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after job"),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    with_sandbox: bool = typer.Option(None, help="Whether to use the sandboxing feature to host the model"),
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
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
):
    """Run a pre-defined module or script in the NeMo-Skills container."""
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'

    # Assert that either command or extra_arguments is provided, not both
    if command and extra_arguments:
        raise ValueError("Please provide either a command or extra arguments, not both.")
    elif not command and not extra_arguments:
        raise ValueError("Please provide either a command or extra arguments.")

    command = command or extra_arguments  # From here on, `command` will be used as the command to run
    extra_arguments = ""  # Reset extra_arguments to avoid confusion

    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(cluster_config, mount_paths, create_remote_dir=check_mounted_paths)

    if log_dir:
        if check_mounted_paths: pipeline_utils.create_remote_directory(log_dir, cluster_config)
        log_dir = pipeline_utils.get_mounted_path(cluster_config, log_dir)

    # Resolve if chunking is necessary
    if num_chunks:
        chunk_ids = utils.compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    # Resolve if we need to use the inference server
    get_random_port = server_gpus != 8 and not exclusive

    with get_exp(expname, cluster_config) as exp:
        for chunk_id in chunk_ids:
            if model is not None:
                server_config, extra_arguments, server_address, server_port = pipeline_utils.configure_client(
                    model=model,
                    server_type=server_type,
                    server_address=server_address,
                    server_gpus=server_gpus,
                    server_nodes=server_nodes,
                    server_args=server_args,
                    extra_arguments=extra_arguments,  # this is empty string by design
                    get_random_port=get_random_port,
                )
            else:
                server_config = None
                server_port = None

            cmd = get_cmd(
                command=command,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
            )

            prev_tasks = None
            for _ in range(dependent_jobs + 1):
                # Add the task to the experiment
                task_name = f"{expname}_chunk_{chunk_id}" if chunk_id is not None else expname
                new_task = add_task(
                    exp,
                    cmd=pipeline_utils.get_generation_command(server_address, cmd),
                    task_name=task_name,
                    log_dir=log_dir,
                    container=cluster_config["containers"][container],
                    cluster_config=cluster_config,
                    partition=partition,
                    time_min=time_min,
                    server_config=server_config,
                    with_sandbox=with_sandbox,
                    sandbox_port=None if get_random_port else 6000,
                    run_after=run_after,
                    reuse_code=reuse_code,
                    reuse_code_exp=reuse_code_exp,
                    task_dependencies=prev_tasks,
                    num_gpus=num_gpus,
                    num_nodes=num_nodes,
                    slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                )
                prev_tasks = [new_task]
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
