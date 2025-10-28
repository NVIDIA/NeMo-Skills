# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
from typing import Dict, List, Optional, Tuple

import typer
from omegaconf import DictConfig, ListConfig, OmegaConf

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
    # Evaluator mapping/config knobs
    latest_mapping: bool = typer.Option(
        False, help="If True, use latest upstream task mapping from nemo_evaluator_launcher"
    ),
    tasks_mapping_toml: Optional[str] = typer.Option(
        None, help="Path to a local mapping.toml to resolve harness/task containers"
    ),
    # Optional per-container installation step before running evaluator
    install_cmd: Optional[str] = typer.Option(
        None,
        help="Shell command to run inside container before evaluator (e.g., pip install -r /workspace/reqs.txt)",
    ),
):
    """Run Nemo Evaluator tasks via nemo-skills orchestration.

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

    # Helpers for container + env resolution
    def _parse_override_flag(args: List[str], key: str) -> Optional[str]:
        prefix = f"++{key}="
        for a in args:
            if a.startswith(prefix):
                return a[len(prefix) :]
        return None

    def _normalize_env_map(value) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if value is None:
            return env

        # Gracefully handle OmegaConf containers

        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_object(value)

        # Mapping form: {KEY: VALUE}
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    continue
                # Accept common scalar types and cast to str
                if isinstance(v, (str, int, float, bool)) or v is None:
                    env[k] = "" if v is None else str(v)
                else:
                    env[k] = str(v)
            return env

        # Single string is ambiguous; require explicit forms
        raise ValueError("env_vars must be a dict")

    # Now preparing the config for the launcher
    # 1)  tasks from CLI
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    # 2) Resolve container per task via launcher mapping
    from nemo_evaluator_launcher.common.mapping import get_task_from_mapping, load_tasks_mapping  # type: ignore

    mapping = load_tasks_mapping(latest=False)

    def _task_to_container(task_query: str) -> str:
        # Accept 'task' or 'harness.task'
        task_def = get_task_from_mapping(task_query, mapping)
        container = task_def.get("container")
        if not container:
            raise ValueError(f"No container specified for task {task_query!r} in nemo_evaluator_launcher's mapping")
        return container

    # 3) Build launcher RunConfig to collect env_vars: use same overrides the evaluator will receive
    from nemo_evaluator_launcher.api import RunConfig  # type: ignore

    cfg_dir = _parse_override_flag(ctx.args, "nemo_eval_config_dir")
    cfg_name = _parse_override_flag(ctx.args, "nemo_eval_config_name") or "config"
    run_cfg = None
    if cfg_dir:
        run_cfg = RunConfig.from_hydra(config_dir=cfg_dir, config_name=cfg_name, hydra_overrides=list(ctx.args))

    # 4) Build per-task env maps (global overlaid by per-task)
    global_env: Dict[str, str] = {}
    per_task_env: Dict[str, Dict[str, str]] = {}
    if run_cfg is not None:
        evaluation = getattr(run_cfg, "evaluation", None)
        if evaluation is not None:
            global_env = _normalize_env_map(getattr(evaluation, "env_vars", None))
            # Build name->task cfg map to fetch per-task envs
            tasks_cfg = getattr(evaluation, "tasks", []) or []
            name_to_task = {getattr(t, "name", None): t for t in tasks_cfg if getattr(t, "name", None)}
            for tq in task_list:
                tname = tq.split(".", 1)[-1]
                tcfg = name_to_task.get(tname)
                env_map = dict(global_env)
                if tcfg is not None:
                    env_map.update(_normalize_env_map(getattr(tcfg, "env_vars", None)))
                per_task_env[tq] = env_map

    # 5) Group tasks by (container, env_signature)
    def _env_signature(env: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(env.items()))

    groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[str]] = {}
    group_envs: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, str]] = {}
    for tq in task_list:
        cont = _task_to_container(tq)
        env_map = per_task_env.get(tq, global_env)
        sig = _env_signature(env_map)
        key = (cont, sig)
        groups.setdefault(key, []).append(tq)
        group_envs[key] = env_map

    # 6) Build jobs per group
    jobs = []
    for idx, ((container_id, sig), group_tasks) in enumerate(groups.items()):
        tasks_arg = ",".join(group_tasks)
        eval_cmd = (
            "export HYDRA_FULL_ERROR=1 && "
            "python -m nemo_skills.inference.nemo_evaluator "
            f"++tasks={tasks_arg} "
            f"{extra_overrides}"
        )

        # Ensure CPU-only runs don't request GPUs from DockerExecutor:
        # When job_gpus == 0, explicitly set metadata.gpus = None so
        # exec_config["num_gpus"] becomes None (not 0), preventing GPU runtime selection.
        client_cmd = Command(
            command=eval_cmd,
            container=container_id,  # key or full image
            gpus=job_gpus or None,
            nodes=job_nodes or 1,
            name=f"{expname}-{idx}" if len(groups) > 1 else expname,
            installation_command=install_cmd,
            metadata={
                "log_prefix": "main",
                "environment": group_envs.get((container_id, sig), {}),
                "gpus": job_gpus or None,
            },
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
            name=f"{expname}-{idx}" if len(groups) > 1 else expname,
            log_dir=log_dir,
        )

        jobs.append({"name": group.name, "group": group})

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
