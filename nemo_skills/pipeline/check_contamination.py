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

import os
from enum import Enum
from typing import List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.generate import wrap_cmd
from nemo_skills.pipeline.utils import (
    add_mount_path,
    add_task,
    check_if_mounted,
    check_mounts,
    create_remote_directory,
    get_cluster_config,
    get_exp,
    get_free_port,
    get_generation_command,
    get_mounted_path,
    is_mounted_filepath,
    resolve_mount_paths,
    run_exp,
)
from nemo_skills.utils import setup_logging


def get_check_contamination_cmd(input_file, output_file, extra_arguments=""):
    cmd = (
        f"python -m nemo_skills.inference.check_contamination "
        f"    ++input_file={input_file} "
        f"    ++output_file={output_file} "
        f"    {extra_arguments} "
    )
    return cmd


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def check_contamination(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_file: str = typer.Option(
        ..., help="Input file with the data to check for contamination. An output of the retrieve_similar.py script."
    ),
    output_file: str = typer.Option(..., help="Where to save results"),
    expname: str = typer.Option("check-contamination", help="Nemo run experiment name"),
    model: str = typer.Option(None, help="Path to the model or model name in API."),
    server_address: str = typer.Option(
        None, help="Use ip:port for self-hosted models or the API url if using model providers."
    ),
    server_type: SupportedServers = typer.Option(SupportedServers.trtllm, help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server."),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server."),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
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
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    preprocess_cmd: str = typer.Option(None, help="Command to run before generation"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after generation"),
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
    """Check contamination between train/test via an LLM call.

    Run `python -m nemo_skills.inference.check_contamination --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    if dependent_jobs > 0:
        extra_arguments += " ++skip_filled=True "
    try:
        server_type = server_type.value
    except AttributeError:
        pass

    get_random_port = server_gpus != 8 and not exclusive

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    input_file, output_file, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={input_file: "/mounted_data/input", output_file: "/mounted_data/output"},
        check_mounted_paths=check_mounted_paths,
    )

    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_port = get_free_port(strategy="random") if get_random_port else 5000
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_entrypoint": server_entrypoint,
            "server_port": server_port,
        }
        extra_arguments += f" ++server.server_type={server_type} "
        extra_arguments += f" ++server.host=localhost "
        extra_arguments += f" ++server.port={server_port} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
        )

    with get_exp(expname, cluster_config) as exp:
        prev_tasks = None
        for _ in range(dependent_jobs + 1):
            new_task = add_task(
                exp,
                cmd=wrap_cmd(
                    get_generation_command(
                        server_address=server_address,
                        generation_commands=get_check_contamination_cmd(input_file, output_file, extra_arguments),
                    ),
                    preprocess_cmd=preprocess_cmd,
                    postprocess_cmd=postprocess_cmd,
                ),
                task_name=expname,
                log_dir=log_dir,
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                server_config=server_config,
                task_dependencies=prev_tasks,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )
            prev_tasks = [new_task]
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
