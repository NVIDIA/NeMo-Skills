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
from enum import Enum

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import add_task, check_if_mounted, get_cluster_config, get_exp, get_free_port
from nemo_skills.utils import setup_logging


def get_gradio_chat_cmd(executor, server_type):
    # if cluster is local, we want to automatically launch an app
    # otherwise, we launch configuration page that asks for server address
    mode = 'direct' if executor == "local" else 'manual'
    cmd = (
        "python -m nemo_skills.inference.chat_interface.launch "
        f"launch_mode={mode} "
        f"server_type={server_type} "
    )
    return cmd


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    sglang = "sglang"


@app.command()
@typer_unpacker
def start_server(
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    model: str = typer.Option(..., help="Path to the model"),
    server_type: SupportedServers = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(..., help="Number of GPUs to use for hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use for hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    with_sandbox: bool = typer.Option(
        False, help="Starts a sandbox (set this flag if model supports calling Python interpreter)"
    ),
    launch_chat_interface: bool = typer.Option(
        False, help="If True, will launch a gradio app that provides chat with the model"
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
    get_random_port: bool = typer.Option(False, help="If True, will get a random port for the server"),
):
    """Self-host a model server."""
    setup_logging(disable_hydra_logs=False, use_rich=True)

    cluster_config = get_cluster_config(cluster, config_dir)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    if log_dir:
        check_if_mounted(cluster_config, log_dir)

    server_config = {
        "model_path": model,
        "server_type": server_type,
        "num_gpus": server_gpus,
        "num_nodes": server_nodes,
        "server_args": server_args,
        "server_port": get_free_port(strategy="random") if get_random_port else 5000,
    }

    with get_exp("server", cluster_config) as exp:
        if launch_chat_interface:
            if server_type in ["trtllm", "nemo"]:
                raise ValueError(f"Currently chat interface for {server_type} is not supported.")
            add_task(
                exp,
                cmd=get_gradio_chat_cmd(cluster_config["executor"], server_type),
                task_name='gradio_chat',
                container="",
                cluster_config=get_cluster_config(), # launch app locally
            )
        add_task(
            exp,
            cmd="",  # not running anything except the server
            task_name='server',
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            time_min=time_min,
            server_config=server_config,
            with_sandbox=with_sandbox,
            sandbox_port=None if get_random_port else 6000,
            slurm_kwargs={"exclusive": exclusive} if exclusive else None,
        )
        # we don't want to detach in this case even on slurm, so not using run_exp
        exp.run(detach=False, tail_logs=True)
        # TODO: seems like not being killed? If nemorun doesn't do this, we can catch the signal and kill the server ourselves


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
