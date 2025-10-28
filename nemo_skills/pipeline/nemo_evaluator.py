# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
from typing import List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def nemo_evaluator(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help=(
            "One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
            "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument."
        ),
    ),
    output_dir: str = typer.Option(..., help="Where to put logs and .done tracking files for the run"),
    expname: str = typer.Option("nemo-evaluator", help="Nemo run experiment name"),
    tasks: str = typer.Option(
        ...,
        help="Comma-separated list of evaluator task queries (`taskname` "
        "or, for the sake of name resolution, `harness.taskname`)",
    ),
    # Orchestration knobs
    job_gpus: int = typer.Option(0, help="GPUs to allocate for the evaluator job (client container)"),
    job_nodes: int = typer.Option(1, help="Nodes to allocate for the evaluator job"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    qos: str = typer.Option(None, help="Slurm QoS"),
    time_min: str = typer.Option(None, help="Slurm time-min"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    log_dir: str = typer.Option(None, help="Custom location for logs"),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help=(
            "If True, will keep the mounts for the sandbox container. Risky; sandbox executes LLM commands and "
            "could potentially lead to data loss."
        ),
    ),
    # Experiment lifecycle and dependencies
    reuse_code: bool = typer.Option(True, help="If True, will reuse code from the provided experiment"),
    reuse_code_exp: str = typer.Option(None, help="If specified, reuse code from this experiment"),
    run_after: List[str] = typer.Option(None, help="List of expnames that must complete before this starts"),
    dependent_jobs: int = typer.Option(0, help="Launch this number of dependent jobs"),
    # Config discovery
    config_dir: str = typer.Option(None, help="Where to search for cluster configs"),
    dry_run: bool = typer.Option(False, help="If True, validate without submitting the job"),
    # Mapping flags
    latest_mapping: bool = typer.Option(False, help="Use the latest evaluator task mapping from remote"),
    tasks_mapping_toml: str = typer.Option(None, help="Path to a local mapping.toml for evaluator tasks"),
):
    """Run Nemo Evaluator tasks via nemo-skills orchestration (no server hosting).

    Extra Hydra overrides for the underlying evaluator generator can be passed positionally and will be
    forwarded to the Python -m invocation, e.g. ++nemo_eval_config_dir=..., ++nemo_eval_config_name=..., etc.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)

    extra_overrides = f"{' '.join(ctx.args)}"
    LOG.info("Starting nemo_evaluator job")
    LOG.info("Extra overrides passed to evaluator: %s", extra_overrides)

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(cluster_config, mount_paths, create_remote_dir=False)

    if not log_dir:
        log_dir = f"{output_dir}/nemo-evaluator-logs"

    # Validate mounts for output dir
    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=False,
    )

    # Build evaluator command (do not start servers)
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    tasks_arg = ",".join(task_list)

    eval_cmd = (
        "export HYDRA_FULL_ERROR=1 && "
        "python -m nemo_skills.inference.nemo_evaluator "
        f"++tasks={tasks_arg} "
        f"{extra_overrides}"
    )

    # Create client command only (no server component)
    client_cmd = Command(
        command=eval_cmd,
        container=cluster_config["containers"].get("nemo-skills", "nemo-skills"),
        gpus=job_gpus or None,
        nodes=job_nodes or 1,
        name=expname,
        metadata={"log_prefix": "main"},
    )

    group = CommandGroup(
        commands=[client_cmd],
        hardware=HardwareConfig(
            partition=partition,
            qos=qos,
            time_min=time_min,
            exclusive=exclusive,
            num_gpus=job_gpus or None,
            num_nodes=job_nodes or 1,
        ),
        name=expname,
        log_dir=log_dir,
    )

    jobs = [{"name": expname, "group": group}]

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=True,  # avoid HF_HOME requirement for this orchestration path
        run_after=run_after,
    )

    # Use sequential for local/none executors
    sequential = True if cluster_config.get("executor") in ["local", "none"] else False

    result = pipeline.run(dry_run=dry_run, sequential=sequential)
    return result


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
