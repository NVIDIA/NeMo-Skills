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
from typing import Any, Callable, Dict, List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import import_from_path
from nemo_skills.inference import GENERATION_MODULE_MAP, GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.cluster import parse_sbatch_kwargs
from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))

# TODO: add num_jobs here for consistency with eval?


def _normalize_models_config(
    model: Optional[str | List[str]],
) -> List[str]:
    """
    Normalize model specification to list.

    Args:
        model: Single model path or list of model paths

    Returns:
        List of model paths

    Raises:
        ValueError: If model is None
    """
    if model is None:
        raise ValueError("Must specify --model")
    if isinstance(model, str):
        return [model]
    return list(model)


def _normalize_parameter(
    param_value: Any,
    num_models: int,
    param_name: str,
) -> List[Any]:
    """
    Normalize a parameter to a per-model list.

    Logic:
    1. If param_value is a list:
       - If length == num_models: use as-is
       - If length == 1: broadcast to all models
       - Otherwise: error
    2. If param_value is scalar: broadcast to all models
    3. Return list of length num_models

    Args:
        param_value: Parameter value (scalar or list)
        num_models: Number of models
        param_name: Name of parameter (for error messages)

    Returns:
        List of parameter values (one per model)

    Raises:
        ValueError: If list length doesn't match num_models

    Examples:
        >>> _normalize_parameter(8, 3, "server_gpus")
        [8, 8, 8]
        >>> _normalize_parameter([8, 16], 2, "server_gpus")
        [8, 16]
        >>> _normalize_parameter([8], 3, "server_gpus")
        [8, 8, 8]
    """
    if not isinstance(param_value, list):
        # Scalar: broadcast
        return [param_value] * num_models

    if len(param_value) == num_models:
        # List matches: use as-is
        return list(param_value)

    if len(param_value) == 1:
        # Single-element list: broadcast
        return param_value * num_models

    raise ValueError(
        f"Parameter {param_name} has {len(param_value)} values but {num_models} models specified. "
        f"Must be 1 value (broadcast) or {num_models} values (per-model)."
    )


def _create_commandgroup_from_config(
    generation_cmd: str,
    server_config: Optional[Dict],
    with_sandbox: bool,
    sandbox_port: Optional[int],
    cluster_config: Dict,
    installation_command: Optional[str],
    get_server_command_fn: Callable,
    partition: Optional[str],
    keep_mounts_for_sandbox: bool,
    task_name: str,
    log_dir: str,
    sbatch_kwargs: Optional[Dict] = None,
) -> CommandGroup:
    """Create a CommandGroup from server_config.

    Component ordering:
    1. Server (if server_config provided)
    2. Client command
    3. Sandbox (if with_sandbox=True)
    """

    components = []

    # Track GPU/node requirements for this group (from server config)
    group_gpus = 0
    group_nodes = 1

    # 1. Add server if server_config is provided
    if server_config is not None and int(server_config["num_gpus"]) > 0:
        server_type = server_config["server_type"]
        # Get container from server_config if provided, otherwise fall back to cluster config
        if "container" in server_config:
            server_container = server_config.pop("container")
        else:
            server_container = cluster_config["containers"][server_type]

        # Call server command builder directly with cluster_config
        cmd, num_tasks = get_server_command_fn(**server_config, cluster_config=cluster_config)

        # Set group GPU/node requirements from server config
        group_gpus = server_config["num_gpus"]
        group_nodes = server_config["num_nodes"]

        # Create metadata dict
        metadata = {
            "num_tasks": num_tasks,
            "gpus": group_gpus,
            "nodes": group_nodes,
            "log_prefix": "server",
        }

        server_cmd = Command(
            command=cmd,
            container=server_container,
            gpus=group_gpus,
            nodes=group_nodes,
            name=task_name,
            metadata=metadata,
        )
        components.append(server_cmd)

    # 2. Add main generation command
    # Note: General cluster config env vars are automatically added by get_env_variables() in get_executor()
    client_env = {}
    if with_sandbox and sandbox_port is not None:
        client_env["NEMO_SKILLS_SANDBOX_PORT"] = str(sandbox_port)

    client_cmd = Command(
        command=generation_cmd,
        container=cluster_config["containers"]["nemo-skills"],
        name=task_name,
        installation_command=installation_command,
        metadata={
            "log_prefix": "main",
            "environment": client_env,
        },
    )
    components.append(client_cmd)

    # 3. Add sandbox if requested
    if with_sandbox:
        # Call sandbox command builder directly with cluster_config
        cmd, metadata = sandbox_command(cluster_config=cluster_config, port=sandbox_port)
        metadata["log_prefix"] = "sandbox"

        sandbox_cmd = Command(
            command=cmd,
            container=cluster_config["containers"]["sandbox"],
            name=task_name,
            metadata=metadata,
        )

        components.append(sandbox_cmd)

    return CommandGroup(
        commands=components,
        hardware=HardwareConfig(
            partition=partition,
            num_gpus=group_gpus,
            num_nodes=group_nodes,
            sbatch_kwargs=sbatch_kwargs,
        ),
        name=task_name,
        log_dir=log_dir,
    )


def _create_job_unified(
    models: List[str],
    server_configs: List[Optional[Dict]],
    generation_cmd: str,
    cluster_config: Dict,
    installation_command: Optional[str],
    get_server_command_fn: Callable,
    with_sandbox: bool,
    sandbox_port: Optional[int],
    partition: Optional[str],
    keep_mounts_for_sandbox: bool,
    task_name: str,
    log_dir: str,
    sbatch_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Create a job for n models (unified for n=1 and n>1).

    Structure:
    - Group 0: Model 0 server + client + (optional sandbox)
    - Group 1: Model 1 server (if n>1)
    - Group N: Model N server (if n>1)

    For n=1, returns a single-element list. The Pipeline automatically
    optimizes single-group lists to use efficient single-group jobs.

    Args:
        models: List of model paths
        server_configs: List of server configurations (one per model, None if not hosting)
        generation_cmd: Command to run the generation client
        cluster_config: Cluster configuration
        installation_command: Installation command to run before client
        get_server_command_fn: Function to build server commands
        with_sandbox: Whether to include sandbox
        sandbox_port: Port for sandbox
        partition: Slurm partition
        keep_mounts_for_sandbox: Whether to keep mounts for sandbox
        task_name: Name for the task
        log_dir: Directory for logs
        sbatch_kwargs: Additional sbatch kwargs (including qos, time_min, exclusive, etc.)

    Returns:
        Job dict with "groups" key (list of CommandGroup objects)
    """

    num_models = len(models)
    groups = []

    # Create groups for each model
    for idx, (model_path, server_config) in enumerate(zip(models, server_configs)):
        components = []

        # Track GPU/node requirements for this group (from server config)
        group_gpus = 0
        group_nodes = 1

        # 1. Add server if needed
        if server_config is not None and int(server_config.get("num_gpus", 0)) > 0:
            server_type = server_config["server_type"]

            # Get container
            if "container" in server_config:
                server_container = server_config.pop("container")
            else:
                server_container = cluster_config["containers"][server_type]

            # Build server command
            # Rename model_path key to match what server commands expect
            server_config_copy = server_config.copy()
            if "model_path" in server_config_copy:
                server_config_copy["model"] = server_config_copy.pop("model_path")
            if "server_port" in server_config_copy:
                server_config_copy["port"] = server_config_copy.pop("server_port")

            cmd, num_tasks = get_server_command_fn(**server_config_copy, cluster_config=cluster_config)

            # Set group GPU/node requirements from server config
            group_gpus = server_config["num_gpus"]
            group_nodes = server_config["num_nodes"]

            metadata = {
                "num_tasks": num_tasks,
                "gpus": group_gpus,
                "nodes": group_nodes,
                "log_prefix": f"server_{idx}" if num_models > 1 else "server",
            }

            server_cmd = Command(
                command=cmd,
                container=server_container,
                gpus=group_gpus,
                nodes=group_nodes,
                name=f"model_{idx}_server" if num_models > 1 else "server",
                metadata=metadata,
            )
            components.append(server_cmd)

        # 2. Group 0 gets the client
        if idx == 0:
            client_env = {}
            if with_sandbox and sandbox_port is not None:
                client_env["NEMO_SKILLS_SANDBOX_PORT"] = str(sandbox_port)

            client_cmd = Command(
                command=generation_cmd,
                container=cluster_config["containers"]["nemo-skills"],
                name="generation_client" if num_models > 1 else task_name,
                installation_command=installation_command,
                metadata={
                    "log_prefix": "main",
                    "environment": client_env,
                },
            )
            components.append(client_cmd)

            # 3. Add sandbox to group 0 if requested
            #   Only Group 0 is needed because we do not currently have a way
            #   to route requests between multiple sandboxes
            if with_sandbox:
                cmd, metadata = sandbox_command(
                    cluster_config=cluster_config,
                    port=sandbox_port,
                )
                metadata["log_prefix"] = "sandbox" if num_models == 1 else "sandbox_0"

                sandbox_cmd = Command(
                    command=cmd,
                    container=cluster_config["containers"]["sandbox"],
                    name="sandbox" if num_models == 1 else "sandbox_0",
                    metadata=metadata,
                )
                components.append(sandbox_cmd)

        # Only create group if it has components (skip empty groups for pre-hosted models)
        if components:
            # Create group with explicitly tracked GPU/node requirements
            # (Client and sandbox components have no GPUs, only server does)
            group = CommandGroup(
                commands=components,
                hardware=HardwareConfig(
                    partition=partition,
                    num_gpus=group_gpus,
                    num_nodes=group_nodes,
                    sbatch_kwargs=sbatch_kwargs,
                ),
                name=f"model_{idx}_group" if num_models > 1 else task_name,
                log_dir=log_dir,
            )
            groups.append(group)

    return {
        "name": task_name,
        "groups": groups,
        "dependencies": None,
    }


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def generate(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_file: str = typer.Option(
        None, help="Path to the input data file. Can either specify input_file or input_dir, but not both. "
    ),
    input_dir: str = typer.Option(
        None,
        help="Path to the input data directory. Can either specify input_file or input_dir, but not both. "
        "If input_file is not provided, will use output-rs{{seed}}.jsonl inside input_dir as input_files. "
        "In this case, the random seed parameter is used both for input and for output files, which "
        "means it's a 1-1 mapping (not 1-num_random_seeds as in the case of input_file).",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("generate", help="Nemo run experiment name"),
    generation_type: GenerationType | None = typer.Option(None, help="Type of generation to perform"),
    generation_module: str = typer.Option(
        None,
        help="Path to the generation module to use. "
        "If not specified, will use the registered generation module for the "
        "generation type (which is required in this case).",
    ),
    model: str | List[str] = typer.Option(
        None, help="Path to the model(s) or model name(s) in API. Single value or list for multi-model generation"
    ),
    server_address: str | List[str] = typer.Option(
        None,
        help="Use ip:port for self-hosted models or the API url if using model providers. "
        "Single value (broadcast) or list (per-model)",
    ),
    server_type: pipeline_utils.SupportedServers | List[pipeline_utils.SupportedServers] = typer.Option(
        ..., help="Type of server to use. Single value (broadcast) or list (per-model)"
    ),
    server_gpus: int | List[int] = typer.Option(
        None, help="Number of GPUs to use if hosting the model. Single value (broadcast) or list (per-model)"
    ),
    server_nodes: int | List[int] = typer.Option(
        1, help="Number of nodes required for hosting LLM server. Single value (broadcast) or list (per-model)"
    ),
    server_args: str | List[str] = typer.Option(
        "", help="Any extra arguments to pass to the server. Single value (broadcast) or list (per-model)"
    ),
    server_entrypoint: str | List[str] = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type. Single value (broadcast) or list (per-model)",
    ),
    server_container: str | List[str] = typer.Option(
        None,
        help="Override container image for the hosted server (if server_gpus is set). Single value (broadcast) or list (per-model)",
    ),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    num_random_seeds: int = typer.Option(
        None, help="Specify if want to run many generations with high temperature for the same input"
    ),
    random_seeds: str = typer.Option(
        None,
        help="List of random seeds to use for generation. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    num_chunks: int = typer.Option(
        None,
        help="Number of chunks to split the dataset into. If None, will not chunk the dataset.",
    ),
    chunk_ids: str = typer.Option(
        None,
        help="List of explicit chunk ids to run. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    preprocess_cmd: str = typer.Option(None, help="Command to run before generation"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after generation"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
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
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    exclusive: bool | None = typer.Option(None, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
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
        "nemo-skills",
        help="Name of the wandb project to sync samples to.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    sbatch_kwargs: str = typer.Option(
        "",
        help="Additional sbatch kwargs to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting generation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    # ===== NORMALIZE MODELS AND PARAMETERS TO LISTS =====
    models_list = _normalize_models_config(model)
    num_models = len(models_list)

    LOG.info(f"Number of models: {num_models}")
    for idx, model_name in enumerate(models_list):
        LOG.info(f"  Model {idx}: {model_name}")

    # Convert server_type enum(s) to string(s)
    def convert_server_type_to_string(server_type):
        return server_type.value if hasattr(server_type, "value") else server_type

    if isinstance(server_type, list):
        server_type = [convert_server_type_to_string(st) for st in server_type]
    else:
        server_type = convert_server_type_to_string(server_type)
    # Normalize all server parameters
    server_types_list = _normalize_parameter(server_type, num_models, "server_type")
    server_gpus_list = _normalize_parameter(server_gpus, num_models, "server_gpus")
    server_nodes_list = _normalize_parameter(server_nodes, num_models, "server_nodes")
    server_args_list = _normalize_parameter(server_args, num_models, "server_args")
    server_entrypoints_list = _normalize_parameter(server_entrypoint, num_models, "server_entrypoint")
    server_containers_list = _normalize_parameter(server_container, num_models, "server_container")

    # Handle server_address (can be None)
    if server_address is not None:
        server_addresses_list = _normalize_parameter(server_address, num_models, "server_address")
    else:
        server_addresses_list = [None] * num_models

    # Validate multi-model requirements
    if num_models > 1:
        if generation_type is None:
            raise ValueError("Multi-model generation requires --generation-type to be specified")

    if log_samples:
        wandb_parameters = {
            "name": wandb_name or expname,
            "project": wandb_project,
            "group": wandb_group,
        }
        validate_wandb_project_name(
            wandb_project=wandb_project,
            wandb_name=wandb_name or expname,
            wandb_group=wandb_group,
        )
    else:
        wandb_parameters = None

    if random_seeds and num_random_seeds:
        raise ValueError("Cannot specify both random_seeds and num_random_seeds")
    if num_random_seeds:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))
    if isinstance(random_seeds, str):
        random_seeds = str_ids_to_list(random_seeds)

    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    if not log_dir:
        log_dir = f"{output_dir}/generation-logs"

    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if generation_module is not None and generation_type is not None:
        raise ValueError("Cannot specify both generation_module and generation_type. ")
    if generation_module is None:
        generation_module = GENERATION_MODULE_MAP[generation_type or GenerationType.generate]

    if generation_module.endswith(".py") or os.sep in generation_module:
        path_suffix = ".py" if not generation_module.endswith(".py") else ""
        generation_task = import_from_path(generation_module + path_suffix)
    else:
        generation_task = importlib.import_module(generation_module)
    if not hasattr(generation_task, "GENERATION_TASK_CLASS"):
        raise ValueError(
            f"Module {generation_module} does not have a GENERATION_TASK_CLASS attribute. "
            "Please provide a valid generation module."
        )
    generation_task = generation_task.GENERATION_TASK_CLASS
    extra_arguments = f"{generation_task.get_generation_default_args()} {extra_arguments}"
    extra_arguments_original = extra_arguments

    # Treat no random seeds as a single None seed to unify the code paths
    if not random_seeds:
        random_seeds = [None]

    remaining_jobs = pipeline_utils.get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir,
        random_seeds=random_seeds,
        chunk_ids=chunk_ids,
        rerun_done=rerun_done,
    )

    if _task_dependencies is None:
        _task_dependencies = []

    # Parse sbatch kwargs
    sbatch_kwargs = parse_sbatch_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    # Build jobs list using declarative interface
    jobs = []
    all_job_names = []

    for seed_idx, (seed, chunk_ids) in enumerate(remaining_jobs.items()):
        if wandb_parameters:
            # no need for chunks as it will run after merging
            wandb_parameters["samples_file"] = pipeline_utils.get_chunked_rs_filename(
                output_dir,
                random_seed=seed,
                chunk_id=None,
            )
        for chunk_id in chunk_ids:
            # ===== UNIFIED PATH: Configure all servers and build unified command =====

            server_configs = []
            server_addresses_resolved = []

            # Configure each server
            for idx in range(num_models):
                # Determine port allocation per server (standard logic works for both single and multi-model)
                get_random_port_for_server = pipeline_utils.should_get_random_port(server_gpus_list[idx], exclusive)

                srv_config, srv_address, _ = pipeline_utils.configure_client(
                    model=models_list[idx],
                    server_type=server_types_list[idx],
                    server_address=server_addresses_list[idx],
                    server_gpus=server_gpus_list[idx],
                    server_nodes=server_nodes_list[idx],
                    server_args=server_args_list[idx],
                    server_entrypoint=server_entrypoints_list[idx],
                    server_container=server_containers_list[idx],
                    extra_arguments="",  # Don't pass extra args to server config
                    get_random_port=get_random_port_for_server,
                )
                server_configs.append(srv_config)
                server_addresses_resolved.append(srv_address)

            # Build generation command (unified for single and multi-model)
            cmd = pipeline_utils.get_generation_cmd(
                input_file=input_file,
                input_dir=input_dir,
                random_seed=seed,
                output_dir=output_dir,
                server_addresses=server_addresses_resolved,
                model_names=models_list,
                num_models=num_models,
                extra_arguments=extra_arguments_original,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
                wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                script=generation_module,
                with_sandbox=with_sandbox,
            )
            cmd = pipeline_utils.wrap_python_path(cmd=cmd)

            # Base task name (shared across all dependent jobs in the chain)
            task_name = f"{expname}-rs{seed}" if seed is not None else expname
            if chunk_id is not None:
                task_name += f"-chunk{chunk_id}"

            # Handle dependent_jobs chain
            dependencies = _task_dependencies.copy() if _task_dependencies else []
            prev_job = None

            for dep_idx in range(dependent_jobs + 1):
                # Allocate sandbox port if needed
                # This must be done BEFORE creating job so client knows the port
                if with_sandbox:
                    current_sandbox_port = get_free_port(strategy="random") if get_random_port_for_server else 6000
                else:
                    current_sandbox_port = None

                job_spec = _create_job_unified(
                    models=models_list,
                    server_configs=[cfg.copy() if cfg else None for cfg in server_configs],
                    generation_cmd=cmd,
                    cluster_config=cluster_config,
                    installation_command=installation_command,
                    get_server_command_fn=generation_task.get_server_command_fn(),
                    with_sandbox=with_sandbox,
                    sandbox_port=current_sandbox_port,
                    partition=partition,
                    keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                    task_name=task_name,
                    log_dir=log_dir,
                    sbatch_kwargs=sbatch_kwargs,
                )

                # Use unique internal job name for dependency tracking, but same task_name
                internal_job_name = f"{task_name}-dep{dep_idx}" if dep_idx > 0 else task_name
                job_spec["name"] = internal_job_name

                # Build dependencies: first job in chain gets external dependencies, rest chain to previous
                if dep_idx == 0:
                    # First job: add run_after if no task_dependencies
                    job_deps = dependencies.copy() if dependencies else []
                    if not dependencies and run_after:
                        run_after_list = run_after if isinstance(run_after, list) else [run_after]
                        job_deps.extend(run_after_list)
                    job_deps = job_deps if job_deps else None
                else:
                    # Subsequent jobs in chain depend on previous job (use job object, not string)
                    job_deps = [prev_job]

                job_spec["dependencies"] = job_deps
                jobs.append(job_spec)
                prev_job = job_spec  # Track for next iteration

                all_job_names.append(internal_job_name)

    # If no jobs to run, return early
    if not jobs:
        return None

    # Create and run pipeline
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=skip_hf_home_check,
    )

    # TODO: remove after https://github.com/NVIDIA-NeMo/Skills/issues/578 is resolved as default will be single job
    sequential = True if cluster_config["executor"] in ["local", "none"] else False

    # Pass _reuse_exp to pipeline.run() to add jobs to existing experiment
    result = pipeline.run(dry_run=dry_run, _reuse_exp=_reuse_exp, sequential=sequential)
    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
