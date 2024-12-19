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
from typing import List

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.generate import wrap_cmd
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_cmd(extra_arguments):
    cmd = f"{extra_arguments} "
    cmd = f"export HYDRA_FULL_ERROR=1 && export PYTHONPATH=$PYTHONPATH:/nemo_run/code && cd /nemo_run/code && {cmd}"
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
    container: str = typer.Option("nemo-skills", help="Container to use for the run"),
    expname: str = typer.Option("script", help="Nemo run experiment name"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    num_gpus: int | None = typer.Option(None, help="Number of GPUs to use"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    preprocess_cmd: str = typer.Option(None, help="Command to run before job"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after job"),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(False, help="If True, will use --exclusive flag for slurm"),
):
    """Run a pre-defined module or script in the NeMo-Skills container."""
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'

    cluster_config = get_cluster_config(cluster, config_dir)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)

    with run.Experiment(expname) as exp:
        add_task(
            exp,
            cmd=wrap_cmd(get_cmd(extra_arguments=extra_arguments), preprocess_cmd, postprocess_cmd),
            task_name=expname,
            log_dir=log_dir,
            container=cluster_config["containers"][container],
            cluster_config=cluster_config,
            partition=partition,
            time_min=time_min,
            run_after=run_after,
            reuse_code_exp=reuse_code_exp,
            num_gpus=num_gpus,
            slurm_kwargs={"exclusive": exclusive} if exclusive else None,
        )
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
