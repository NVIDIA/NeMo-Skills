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
import importlib
import logging
import os
from copy import deepcopy
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import ExtraDatasetType
from nemo_skills.evaluation.utils import get_eval_group
from nemo_skills.inference.generate import GenerationTask
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.eval import eval as _eval
from nemo_skills.pipeline.generate import generate as _generate
from nemo_skills.pipeline.run_cmd import run_cmd as _run_cmd
from nemo_skills.utils import compute_chunk_ids, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def prepare_judge_args(prompt_config: str):
    judge_ctx = typer.Context("")
    judge_ctx.args = [f"++prompt_config={prompt_config}"]
    return judge_ctx


def prepare_benchmark_args(
    ctx: typer.Context,
    name: str,
    repeats: int = 1,
    prompt_config: str | None = None,
    inference: dict | None = None,
    judge: dict | None = None,
):
    benchmark = name
    if repeats > 1:
        benchmark = f"{name}:{repeats}"
    ctx = deepcopy(ctx)
    if prompt_config is not None:
        ctx.args.append(f"++prompt_config={prompt_config}")
    if inference is not None:
        for key, value in inference.items():
            ctx.args.append(f"++inference.{key}={value}")

    if judge is not None:
        judge_ctx = prepare_judge_args(**judge)
    else:
        judge_ctx = typer.Context("")
    return benchmark, ctx, judge_ctx


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def eval_group(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to store evaluation results"),
    data_dir: str = typer.Option(
        None,
        help="Path to the data directory. If not specified, will use the default nemo_skills/dataset path. "
        "Can also specify through NEMO_SKILLS_DATA_DIR environment variable.",
    ),
    eval_config: str = typer.Option(
        ...,
        help="Config for the evaluation group to run. "
        "By default searching yaml files inside nemo_skills/evaluation/eval_group, "
        "but can provide an absolute path to a yaml file or a dict with the config directly.",
    ),
    expname: str = typer.Option("eval_group", help="Name of the experiment"),
    model: str = typer.Option(None, help="Path to the model to be evaluated"),
    server_address: str = typer.Option(None, help="Address of the server hosting the model"),
    server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    judge_model: str = typer.Option(None, help="Path to the model to be used as a judge (if applicable)"),
    judge_server_address: str = typer.Option(None, help="Address of the server hosting the judge model"),
    judge_server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Type of server to use for the judge"),
    judge_server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the judge model"),
    judge_server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the judge model"),
    judge_server_args: str = typer.Option("", help="Additional arguments for the judge server"),
    judge_server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the judge server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    num_jobs: int = typer.Option(
        None, help="Number of jobs to split the evaluation into. By default will run all benchmarks/seeds in parallel."
    ),
    num_chunks: int = typer.Option(
        None,
        help="Number of chunks to split the dataset into. If None, will not chunk the dataset.",
    ),
    chunk_ids: str = typer.Option(
        None,
        help="List of explicit chunk ids to run. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    extra_datasets: str = typer.Option(
        None,
        help="Path to a custom dataset folder that will be searched in addition to the main one. "
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS.",
    ),
    extra_datasets_type: ExtraDatasetType = typer.Option(
        "local",
        envvar="NEMO_SKILLS_EXTRA_DATASETS_TYPE",
        help="If you have extra datasets locally, set to 'local', if on cluster, set to 'cluster'."
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS_TYPE environment variable.",
    ),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    log_samples: bool = typer.Option(
        False,
        help="If True, will log random samples from the output files to wandb. "
        "Requires WANDB_API_KEY to be set in the environment. "
        "Use wandb_name/wandb_group/wandb_project to specify where to log.",
    ),
    wandb_name: str = typer.Option(
        None,
        help="Name of the wandb group to sync samples to. If not specified, but log_samples=True, will use expname.",
    ),
    wandb_group: str = typer.Option(None, help="Name of the wandb group to sync samples to."),
    wandb_project: str = typer.Option(
        'nemo-skills',
        help="Name of the wandb project to sync samples to.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
):
    """Evaluate a model using a benchmark group config.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting evaluation group job")
    LOG.info("Extra arguments that will be passed to the underlying scripts: %s", extra_arguments)

    default_eval_args = {
        'cluster': cluster,
        'output_dir': output_dir,
        'data_dir': data_dir,
        'model': model,
        'server_address': server_address,
        'server_type': server_type,
        'server_gpus': server_gpus,
        'server_nodes': server_nodes,
        'server_args': server_args,
        'server_entrypoint': server_entrypoint,
        'dependent_jobs': dependent_jobs,
        'starting_seed': starting_seed,
        'num_jobs': num_jobs,
        'num_chunks': num_chunks,
        'chunk_ids': chunk_ids,
        'partition': partition,
        'time_min': time_min,
        'mount_paths': mount_paths,
        'run_after': run_after,
        'reuse_code_exp': reuse_code_exp,
        'reuse_code': reuse_code,
        'config_dir': config_dir,
        'extra_datasets': extra_datasets,
        'extra_datasets_type': extra_datasets_type,
        'exclusive': exclusive,
        'rerun_done': rerun_done,
        'with_sandbox': with_sandbox,
        'check_mounted_paths': check_mounted_paths,
        'wandb_group': wandb_group,
        'wandb_project': wandb_project,
        'installation_command': installation_command,
    }
    default_judge_args = {
        'cluster': cluster,
        'output_dir': output_dir.rstrip('/') + '/judged-results',
        'model': judge_model,
        'server_address': judge_server_address,
        'server_type': judge_server_type,
        'server_gpus': judge_server_gpus,
        'server_nodes': judge_server_nodes,
        'server_args': judge_server_args,
        'server_entrypoint': judge_server_entrypoint,
        'partition': partition,
        'time_min': time_min,
        'mount_paths': mount_paths,
        'reuse_code_exp': reuse_code_exp,
        'reuse_code': reuse_code,
        'config_dir': config_dir,
        'exclusive': exclusive,
        'rerun_done': rerun_done,
        'check_mounted_paths': check_mounted_paths,
        'installation_command': installation_command,
    }

    if log_dir is None:
        log_dir = f"{output_dir}/eval-logs"

    eval_group = get_eval_group(eval_config)
    for job_idx, job_config in enumerate(eval_group['jobs']):
        job_name = job_config.pop('name', str(job_idx))
        job_extra_arguments = job_config.pop('wrap_arguments', None)
        judge_config = job_config.pop('judge', None)

        job_args = deepcopy(default_eval_args)
        job_args['expname'] = f"{expname}-{job_name}"
        job_args['log_dir'] = f"{log_dir}/{job_name}"
        job_args['wandb_name'] = f"{wandb_name}-{job_name}"
        job_args.update(job_config)
        job_ctx = deepcopy(ctx)
        if job_extra_arguments:
            job_ctx.args.extend(job_extra_arguments.split(" "))
        _eval(ctx=job_ctx, **job_args)
        if judge_config:
            judge_args = deepcopy(default_judge_args)
            judge_args['expname'] = f"{expname}-{job_name}-judge"
            judge_args['log_dir'] = f"{log_dir}/{job_name}-judge"
            judge_extra_arguments = judge_config.pop('wrap_arguments', None)
            judge_args.update(judge_config)
            judge_ctx = typer.Context("")
            if judge_extra_arguments:
                judge_ctx.args.extend(judge_extra_arguments.split(" "))
            _generate(ctx=judge_ctx, run_after=job_args['expname'], **judge_args)

        # TODO summarize results and final aggregate metrics


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
