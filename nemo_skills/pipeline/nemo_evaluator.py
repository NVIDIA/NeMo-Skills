# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
import os
from typing import Dict, List, Optional, Tuple

import typer
from omegaconf import DictConfig, ListConfig, OmegaConf

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.commands import sandbox_command, vllm_server_command
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
    # Optional self-hosted server (declarative) ? if server_type is set, a server will be co-scheduled
    server_type: Optional[str] = typer.Option(
        None,
        help=("If set, self-host a server and co-schedule with evaluator. Supported values: vllm (preferred)."),
    ),
    server_model: Optional[str] = typer.Option(
        None, help="Model path/name to serve when self-hosting (e.g., Qwen/Qwen3-4B-Thinking-2507)"
    ),
    server_gpus: int = typer.Option(0, help="GPUs to allocate for the self-hosted server (0 = no server)"),
    server_nodes: int = typer.Option(1, help="Nodes to allocate for the self-hosted server"),
    server_port: Optional[int] = typer.Option(None, help="Port for the server; if unset uses free/random"),
    server_args: Optional[str] = typer.Option(None, help="Extra args for server (passed through)"),
    server_entrypoint: Optional[str] = typer.Option(None, help="Custom entrypoint for server (advanced)"),
    server_container: Optional[str] = typer.Option(
        None, help="Container key/image for server (defaults to cluster 'nemo-skills' if None)"
    ),
    server_base_url: Optional[str] = typer.Option(
        None, help="Use an externally hosted server instead of self-hosting (e.g., http://host:port)"
    ),
    server_api_path: str = typer.Option("/v1/chat/completions", help="API path used for evaluator target url"),
    server_health_path: str = typer.Option("/health", help="Health path used to wait for server readiness"),
    # Optional judge self-hosted server (similar to main server)
    judge_server_type: Optional[str] = typer.Option(
        None,
        help=("If set, self-host a judge server and co-schedule with evaluator. Supported values: vllm (preferred)."),
    ),
    judge_server_model: Optional[str] = typer.Option(
        None, help="Model path/name to serve for judge (e.g., Qwen/Qwen3-32B-Instruct)"
    ),
    judge_server_gpus: int = typer.Option(0, help="GPUs to allocate for the judge server (0 = no judge server)"),
    judge_server_nodes: int = typer.Option(1, help="Nodes to allocate for the judge server"),
    judge_server_port: Optional[int] = typer.Option(None, help="Port for the judge server; if unset uses free/random"),
    judge_server_args: Optional[str] = typer.Option(None, help="Extra args for judge server (passed through)"),
    judge_server_entrypoint: Optional[str] = typer.Option(None, help="Custom entrypoint for judge server (advanced)"),
    judge_server_container: Optional[str] = typer.Option(
        None, help="Container key/image for judge server (defaults to cluster 'vllm' or 'nemo-skills')"
    ),
    judge_server_base_url: Optional[str] = typer.Option(
        None, help="Use an externally hosted judge server instead of self-hosting (e.g., http://host:port)"
    ),
    judge_server_api_path: str = typer.Option("/v1/chat/completions", help="API path used for judge target url"),
    judge_server_health_path: str = typer.Option(
        "/health", help="Health path used to wait for judge server readiness"
    ),
    # Generic launcher override mechanisms
    launcher_overlay: Optional[str] = typer.Option(
        None,
        help=(
            "Overlay for the evaluator launcher Hydra config; accepts a path to YAML/JSON "
            "or an inline YAML/JSON string. Flattened into ++key=value overrides unless explicitly set in ctx.args."
        ),
    ),
    launcher_override: List[str] = typer.Option(
        None,
        help=(
            "Extra Hydra overrides for the evaluator launcher as key=value pairs. "
            "Provide multiple --launcher-override flags to add more entries."
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
):
    """Run Nemo Evaluator tasks via nemo-skills orchestration.

    Extra Hydra overrides for the evaluator launcher config can be passed positionally, e.g.
    ++nemo_eval_config_dir=..., ++nemo_eval_config_name=..., etc.
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
    # 1) Resolve container mapping utilities
    from nemo_evaluator_launcher.common.mapping import get_task_from_mapping, load_tasks_mapping  # type: ignore

    mapping = load_tasks_mapping(latest=latest_mapping)

    def _task_to_container(task_query: str) -> str:
        # Accept 'task' or 'harness.task'
        task_def = get_task_from_mapping(task_query, mapping)
        container = task_def.get("container")
        if not container:
            raise ValueError(f"No container specified for task {task_query!r} in nemo_evaluator_launcher's mapping")
        # WIPP
        return "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_dle/users/agronskiy/images/safety-25.10.sqsh"
        return container

    # 2) Build launcher RunConfig to collect env_vars and to construct final commands on the client
    from nemo_evaluator_launcher.api import RunConfig  # type: ignore

    # Generic override builder: start with ctx.args and extend with overlay/mapping
    def _extract_set_keys(args: List[str]) -> set[str]:
        keys: set[str] = set()
        for a in args:
            if a.startswith("++") and "=" in a:
                k = a[2:].split("=", 1)[0]
                keys.add(k)
        return keys

    def _flatten(prefix: str, obj, out: Dict[str, str]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                _flatten(f"{prefix}.{idx}" if prefix else str(idx), v, out)
        else:
            out[prefix] = "" if obj is None else str(obj)

    def _build_hydra_overrides() -> List[str]:
        overrides: List[str] = list(ctx.args)
        already = _extract_set_keys(overrides)

        # 2a) explicit repeated key=value flags
        if launcher_override:
            for pair in launcher_override:
                if not pair or "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                if k not in already:
                    overrides.append(f"++{k}={v}")
                    already.add(k)

        # 2b) overlay file or inline YAML/JSON
        if launcher_overlay:
            try:
                if os.path.exists(launcher_overlay):
                    ov_cfg = OmegaConf.load(launcher_overlay)
                else:
                    ov_cfg = OmegaConf.create(launcher_overlay)
                ov_dict = OmegaConf.to_container(ov_cfg, resolve=True) or {}
                flat: Dict[str, str] = {}
                _flatten("", ov_dict, flat)
                for k, v in flat.items():
                    if k not in already:
                        overrides.append(f"++{k}={v}")
                        already.add(k)
            except Exception as e:
                LOG.warning("Failed to parse launcher_overlay; ignoring", exc_info=e)

        # 2c) declarative mapping from selected top-level options
        def _add_if_missing(key: str, value: Optional[str]):
            if value is None:
                return
            if key in already:
                return
            overrides.append(f"++{key}={value}")
            already.add(key)

        # infer external vs hosted server
        hosting_server = bool(server_type) and (server_gpus or 0) > 0 and bool(server_model)
        with_external_server = (not hosting_server) and bool(server_base_url)

        # model_id mapping
        _add_if_missing("target.api_endpoint.model_id", server_model)

        # external URL mapping
        if with_external_server and server_base_url:
            url = server_base_url.rstrip("/") + server_api_path
            _add_if_missing("target.api_endpoint.url", url)

        return overrides

    merged_overrides = _build_hydra_overrides()

    cfg_dir = _parse_override_flag(merged_overrides, "nemo_eval_config_dir")
    cfg_name = _parse_override_flag(merged_overrides, "nemo_eval_config_name") or "config"
    if not cfg_dir:
        raise ValueError("++nemo_eval_config_dir is required to construct evaluator commands on the client")

    run_cfg = RunConfig.from_hydra(
        config_dir=cfg_dir,
        config_name=cfg_name,
        hydra_overrides=merged_overrides,
    )

    # 3) Determine tasks to run from the evaluator config
    evaluation = getattr(run_cfg, "evaluation", None)
    if evaluation is None:
        raise ValueError("Evaluator config missing 'evaluation' section with tasks")
    tasks_cfg = getattr(evaluation, "tasks", []) or []
    task_list = [getattr(t, "name", None) for t in tasks_cfg]
    task_list = [t for t in task_list if t]
    if not task_list:
        raise ValueError("No tasks found in evaluator config; please define evaluation.tasks")

    # 4) Build per-task env maps (global overlaid by per-task)
    global_env: Dict[str, str] = {}
    per_task_env: Dict[str, Dict[str, str]] = {}
    name_to_task = {getattr(t, "name", None): t for t in tasks_cfg if getattr(t, "name", None)}
    name_to_index = {getattr(t, "name", None): i for i, t in enumerate(tasks_cfg) if getattr(t, "name", None)}
    global_env = _normalize_env_map(getattr(evaluation, "env_vars", None))
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
        # Helper to construct final evaluator commands on the client
        from nemo_evaluator_launcher.common.helpers import get_eval_factory_command  # type: ignore

        def _build_task_cmd(
            task_query: str,
            url_override: Optional[str] = None,
            judge_url_override: Optional[str] = None,
            judge_model_id: Optional[str] = None,
        ) -> str:
            # Map 'task' or 'harness.task' to task cfg
            task_name = task_query.split(".", 1)[-1]
            task_cfg = name_to_task.get(task_name)
            task_def = get_task_from_mapping(task_query, mapping)

            # Prefer setting URL directly on RunConfig prior to command construction
            url_set = False
            if url_override:
                # If URL contains shell refs like $(scontrol ...), avoid baking into config; pass via --overrides
                contains_shell_ref = ("$(" in url_override) or ("$SLURM_" in url_override)
                if not contains_shell_ref:
                    try:
                        api_ep = getattr(getattr(run_cfg, "target", None), "api_endpoint", None)
                        if api_ep is not None and hasattr(api_ep, "url"):
                            setattr(api_ep, "url", url_override)
                            url_set = True
                    except Exception:
                        # Best-effort; if structure differs, fall back to CLI override
                        url_set = False

            cmd_struct = get_eval_factory_command(run_cfg, task_cfg, task_def)

            # Fallback: if we could not set URL on RunConfig, or need task-scoped judge overrides,
            # append via launcher-style --overrides
            override_parts: List[str] = []
            if url_override and not url_set:
                override_parts.append(f"target.api_endpoint.url={url_override}")

            if judge_url_override or judge_model_id:
                if judge_url_override:
                    override_parts.append(f"config.params.extra.judge.url={judge_url_override}")
                if judge_model_id:
                    override_parts.append(f"config.params.extra.judge.model_id={judge_model_id}")

            if override_parts:
                joined = ",".join(override_parts)
                return f'{cmd_struct.cmd} --overrides "{joined}"'

            return cmd_struct.cmd

        shared_env = group_envs.get((container_id, sig), {})

        commands: List[Command] = []

        # Optional self-hosted servers
        hosting_server = bool(server_type) and (server_gpus or 0) > 0 and bool(server_model)
        with_external_server = (not hosting_server) and bool(server_base_url)
        hosting_judge = bool(judge_server_type) and (judge_server_gpus or 0) > 0 and bool(judge_server_model)
        with_external_judge = (not hosting_judge) and bool(judge_server_base_url)

        # Both can be hosted in one job; we will wait on both and inject URLs per task

        server_command_obj: Optional[Command] = None
        judge_server_command_obj: Optional[Command] = None
        server_effective_port: Optional[int] = None
        judge_effective_port: Optional[int] = None

        if hosting_server:
            stype = (server_type or "vllm").lower()
            sargs = server_args or ""

            if stype != "vllm":
                LOG.warning("Only vllm server_type is explicitly supported in this path right now; got %s", stype)

            # Build server command + metadata (port, num_tasks)
            srv_cmd_str, srv_meta = vllm_server_command(
                cluster_config=cluster_config,
                model=server_model,  # type: ignore[arg-type]
                port=server_port,
                server_type=stype,
                gpus=server_gpus,
                nodes=server_nodes,
                args=sargs,
                entrypoint=server_entrypoint,
            )

            server_effective_port = int(srv_meta.get("port")) if srv_meta and srv_meta.get("port") else None
            server_type = server_type or "vllm"
            # get container from server_config if provided, otherwise fall back to cluster config
            if not server_container:
                server_container = cluster_config["containers"][server_type]
            server_command_obj = Command(
                command=srv_cmd_str,
                container=server_container,
                gpus=server_gpus,
                nodes=server_nodes or 1,
                name=f"{expname}-server-{idx}" if len(groups) > 1 else f"{expname}-server",
                metadata={
                    **srv_meta,
                    "gpus": server_gpus,
                    "log_prefix": "server",
                },
            )
            commands.append(server_command_obj)

        if hosting_judge:
            jstype = (judge_server_type or "vllm").lower()
            jargs = judge_server_args or ""
            if jstype != "vllm":
                LOG.warning(
                    "Only vllm judge_server_type is explicitly supported in this path right now; got %s", jstype
                )
            j_cmd_str, j_meta = vllm_server_command(
                cluster_config=cluster_config,
                model=judge_server_model,  # type: ignore[arg-type]
                port=judge_server_port,
                server_type=jstype,
                gpus=judge_server_gpus,
                nodes=judge_server_nodes,
                args=jargs,
                entrypoint=judge_server_entrypoint,
            )
            if not judge_server_container:
                judge_server_container = cluster_config["containers"].get("vllm") or cluster_config["containers"].get(
                    "nemo-skills", "nemo-skills"
                )
            judge_server_command_obj = Command(
                command=j_cmd_str,
                container=judge_server_container,
                gpus=judge_server_gpus,
                nodes=judge_server_nodes or 1,
                name=f"{expname}-judge-server-{idx}" if len(groups) > 1 else f"{expname}-judge-server",
                metadata={
                    **j_meta,
                    "gpus": judge_server_gpus,
                    "log_prefix": "judge-server",
                },
            )
            commands.append(judge_server_command_obj)

        # Build client command factory that can wait on hosted servers and inject runtime URLs
        if (hosting_server and server_command_obj is not None) or (
            hosting_judge and judge_server_command_obj is not None
        ):

            def _client_cmd_factory():
                # Cross-component references resolved at runtime
                waits: List[str] = []
                target_url: Optional[str] = None
                judge_url: Optional[str] = None

                if hosting_server and server_command_obj is not None:
                    server_host = server_command_obj.hostname_ref()
                    server_port_val = server_command_obj.meta_ref("port")
                    base_url = f"http://{server_host}:{server_port_val}"
                    health_url = f"{base_url}{server_health_path}"
                    target_url = f"{base_url}{server_api_path}"
                    waits.append(pipeline_utils.get_server_wait_cmd(health_url))

                if hosting_judge and judge_server_command_obj is not None:
                    jhost = judge_server_command_obj.hostname_ref()
                    jport = judge_server_command_obj.meta_ref("port")
                    jbase = f"http://{jhost}:{jport}"
                    jhealth = f"{jbase}{judge_server_health_path}"
                    judge_url = f"{jbase}{judge_server_api_path}"
                    waits.append(pipeline_utils.get_server_wait_cmd(jhealth))

                wait_cmd = " && ".join(waits) if waits else "true"
                # Build per-task commands with runtime URL override(s)
                cmds = [
                    _build_task_cmd(
                        tq,
                        url_override=target_url,
                        judge_url_override=judge_url,
                        judge_model_id=judge_server_model,
                    )
                    for tq in group_tasks
                ]
                joined = " && ".join(cmds)
                return f"{wait_cmd} && {joined}"

            client_cmd = Command(
                command=_client_cmd_factory,
                container=container_id,
                gpus=job_gpus or None,
                nodes=job_nodes or 1,
                name=f"{expname}-{idx}" if len(groups) > 1 else expname,
                metadata={
                    "log_prefix": "main",
                    "environment": shared_env,
                    "gpus": job_gpus or None,
                },
            )

            # If both servers are hosted, prefer a heterogeneous job with separate groups for each server and the client
            if (
                hosting_server
                and hosting_judge
                and server_command_obj is not None
                and judge_server_command_obj is not None
            ):
                # Group 0: main server + client (client waits on both servers and injects URLs)
                server_group = CommandGroup(
                    commands=[server_command_obj, client_cmd],
                    hardware=HardwareConfig(
                        partition=partition,
                        qos=qos,
                        time_min=time_min,
                        exclusive=exclusive,
                        num_gpus=server_gpus or None,
                        num_nodes=server_nodes or 1,
                    ),
                    name=f"{expname}-server-{idx}" if len(groups) > 1 else f"{expname}-server",
                    log_dir=log_dir,
                )
                # Group 1: judge server only
                judge_group = CommandGroup(
                    commands=[judge_server_command_obj],
                    hardware=HardwareConfig(
                        partition=partition,
                        qos=qos,
                        time_min=time_min,
                        exclusive=exclusive,
                        num_gpus=judge_server_gpus or None,
                        num_nodes=judge_server_nodes or 1,
                    ),
                    name=f"{expname}-judge-server-{idx}" if len(groups) > 1 else f"{expname}-judge-server",
                    log_dir=log_dir,
                )
                hetero_groups = [server_group, judge_group]

                # Optional sandbox as separate group (0 GPUs)
                if with_sandbox:
                    from nemo_skills.pipeline.utils.server import get_free_port

                    sandbox_port = get_free_port(strategy="random")
                    sb_cmd_str, sb_meta = sandbox_command(cluster_config, port=sandbox_port)
                    sandbox_group = CommandGroup(
                        commands=[
                            Command(
                                command=sb_cmd_str,
                                container=cluster_config["containers"].get("nemo-skills", "nemo-skills"),
                                gpus=None,
                                nodes=1,
                                name=f"{expname}-sandbox-{idx}" if len(groups) > 1 else f"{expname}-sandbox",
                                metadata=sb_meta,
                            )
                        ],
                        hardware=HardwareConfig(
                            partition=partition,
                            qos=qos,
                            time_min=time_min,
                            exclusive=exclusive,
                            num_gpus=None,
                            num_nodes=1,
                        ),
                        name=f"{expname}-sandbox-{idx}" if len(groups) > 1 else f"{expname}-sandbox",
                        log_dir=log_dir,
                    )
                    hetero_groups.append(sandbox_group)

                jobs.append(
                    {
                        "name": f"{expname}-{idx}" if len(groups) > 1 else expname,
                        "groups": hetero_groups,
                    }
                )
                # Proceed to next container+env group
                continue
        else:
            # Either external server provided, or client-only (no URL override)
            if with_external_server:
                # Ensure trailing slash handling is consistent
                url = server_base_url.rstrip("/") + server_api_path
                # Prefer setting the URL on the RunConfig before building commands
                try:
                    if getattr(getattr(run_cfg, "target", None), "api_endpoint", None):
                        run_cfg.target.api_endpoint.url = url
                except Exception:
                    # Fallback handled below by appending overrides (not needed in most cases)
                    pass
                # External judge handling
                judge_url = None
                if with_external_judge and judge_server_base_url:
                    judge_url = judge_server_base_url.rstrip("/") + judge_server_api_path
                cmds = [
                    _build_task_cmd(
                        tq,
                        judge_url_override=judge_url,
                        judge_model_id=judge_server_model,
                    )
                    for tq in group_tasks
                ]
            else:
                judge_url = None
                if with_external_judge and judge_server_base_url:
                    judge_url = judge_server_base_url.rstrip("/") + judge_server_api_path
                cmds = [
                    _build_task_cmd(
                        tq,
                        judge_url_override=judge_url,
                        judge_model_id=judge_server_model,
                    )
                    for tq in group_tasks
                ]

            eval_cmd = " && ".join(cmds)

            client_cmd = Command(
                command=eval_cmd,
                container=container_id,
                gpus=job_gpus or None,
                nodes=job_nodes or 1,
                name=f"{expname}-{idx}" if len(groups) > 1 else expname,
                metadata={
                    "log_prefix": "main",
                    "environment": shared_env,
                    "gpus": job_gpus or None,
                },
            )

        commands.append(client_cmd)

        # Optional sandbox container in the same job
        if with_sandbox:
            # Use a random port per group; sandbox communicates via env var on client side (set by evaluator harness)
            from nemo_skills.pipeline.utils.server import get_free_port

            sandbox_port = get_free_port(strategy="random")
            sb_cmd_str, sb_meta = sandbox_command(cluster_config, port=sandbox_port)
            sandbox_cmd = Command(
                command=sb_cmd_str,
                container=cluster_config["containers"].get("nemo-skills", "nemo-skills"),
                gpus=None,
                nodes=1,
                name=f"{expname}-sandbox-{idx}" if len(groups) > 1 else f"{expname}-sandbox",
                metadata=sb_meta,
            )
            commands.append(sandbox_cmd)

        # Group hardware: allocate enough GPUs/nodes for all hosted servers
        if hosting_server or hosting_judge:
            total_server_gpus = (server_gpus or 0) + (judge_server_gpus or 0)
            group_num_gpus = total_server_gpus or None
            group_num_nodes = max(server_nodes or 1, judge_server_nodes or 1, job_nodes or 1)
        else:
            group_num_gpus = job_gpus or None
            group_num_nodes = job_nodes or 1

        group = CommandGroup(
            commands=commands,
            hardware=HardwareConfig(
                partition=partition,
                qos=qos,
                time_min=time_min,
                exclusive=exclusive,
                num_gpus=group_num_gpus,
                num_nodes=group_num_nodes,
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
        with_ray=bool(server_type) and (server_nodes or 1) > 1,
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
