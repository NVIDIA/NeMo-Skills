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

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging


def get_check_contamination_cmd(input_file, output_file, extra_arguments=""):
    return (
        f"python -m nemo_skills.inference.check_contamination "
        f"    ++input_file={input_file} "
        f"    ++output_file={output_file} "
        f"    {extra_arguments} "
    )


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
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs"),
    input_file: str = typer.Option(
        ..., help="Input file with the data to check for contamination. An output of the retrieve_similar.py script."
    ),
    output_file: str = typer.Option(..., help="Where to save results"),
    expname: str = typer.Option("llm-math-judge", help="Nemo run experiment name"),
    model: str = typer.Option(None, help="Path to the model or model name in API."),
    server_address: str = typer.Option(
        None, help="Use ip:port for self-hosted models or the API url if using model providers."
    ),
    server_type: SupportedServers = typer.Option(SupportedServers.trtllm, help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server."),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server."),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    run_after: str = typer.Option(
        None,
        help="Can specify an expname that needs to be completed before this one starts (will use as slurm dependency)",
    ),
):
    """Check contamination between train/test via an LLM call."""
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, input_file)
    check_if_mounted(cluster_config, output_file)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)

    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = "localhost:5000"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
        }
        extra_arguments += f" ++server.server_type={server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
        )

    with run.Experiment(expname) as exp:
        add_task(
            exp,
            cmd=get_generation_command(
                server_address=server_address,
                generation_commands=get_check_contamination_cmd(input_file, output_file, extra_arguments),
            ),
            task_name="check-contamination",
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            server_config=server_config,
            run_after=run_after,
        )
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
