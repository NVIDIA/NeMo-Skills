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
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import import_from_path
from nemo_skills.inference import GENERATION_MODULE_MAP, GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.task_factories import PipelineTemplates, TaskFactory
from nemo_skills.pipeline.utils.task_system import PipelineBuilder, TaskGroup
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
)

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def generate_v2(
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
    model: str = typer.Option(None, help="Path to the model or model name in API"),
    server_address: str = typer.Option(
        None, help="Use ip:port for self-hosted models or the API url if using model providers"
    ),
    server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    server_container: str = typer.Option(
        None, help="Override container image for the hosted server (if server_gpus is set)"
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
    with_sandbox: bool = typer.Option(False, help="Whether to run the sandbox"),
    get_random_port: bool = typer.Option(False, help="Whether to get a random port for the sandbox"),
    config_dir: str = typer.Option(None, help="Path to directory with cluster configs"),
    log_dir: str = typer.Option(None, help="Where to put logs"),
    partition: str = typer.Option(None, help="Slurm partition to use"),
    time_min: str = typer.Option(None, help="Minimum time for the job"),
    exclusive: bool = typer.Option(False, help="Whether to request exclusive access to nodes"),
    run_after: str = typer.Option(None, help="Run after this experiment"),
    reuse_code: bool = typer.Option(True, help="Whether to reuse code from previous experiments"),
    reuse_code_exp: str = typer.Option(None, help="Experiment to reuse code from"),
    installation_command: str = typer.Option(None, help="Command to install packages"),
    skip_hf_home_check: bool = typer.Option(False, help="Skip HuggingFace home check"),
    dry_run: bool = typer.Option(False, help="Whether to run in dry-run mode"),
    rerun_done: bool = typer.Option(False, help="Whether to rerun completed jobs"),
    check_mounted_paths: bool = typer.Option(True, help="Whether to check if paths are mounted"),
    preprocess_cmd: str = typer.Option(None, help="Command to run before generation"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after generation"),
    wandb_parameters: str = typer.Option(None, help="WandB parameters for logging"),
    eval_args: str = typer.Option("", help="Arguments for evaluation"),
    extra_arguments: str = typer.Option("", help="Extra arguments to pass to the generation script"),
    _task_dependencies: List[str] = typer.Option(None, help="Internal task dependencies"),
    _reuse_exp=None,  # Internal parameter for reusing experiments
):
    """Generate outputs using the new task system (v2)."""

    # Setup logging
    setup_logging()

    # Parse parameters
    if isinstance(random_seeds, str):
        random_seeds = str_ids_to_list(random_seeds)
    if num_random_seeds is not None and random_seeds is not None:
        raise ValueError("Cannot specify both num_random_seeds and random_seeds")
    if num_random_seeds is not None:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))

    if isinstance(chunk_ids, str):
        chunk_ids = str_ids_to_list(chunk_ids)
    elif chunk_ids is None and num_chunks is not None:
        chunk_ids = compute_chunk_ids(num_chunks)
    else:
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

    original_server_address = server_address

    # Setup generation module
    if generation_module is not None and generation_type is not None:
        raise ValueError("Cannot specify both generation_module and generation_type. ")
    if generation_module is None:
        generation_module = GENERATION_MODULE_MAP[generation_type or GenerationType.generate]

    if os.sep in generation_module:
        generation_task = import_from_path(generation_module)
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

    # Create pipeline builder
    builder = PipelineBuilder(expname, cluster_config)
    builder.set_global_config(
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
    )

    has_tasks = False
    all_task_groups = []

    if _task_dependencies is None:
        _task_dependencies = []

    # Process each seed and chunk combination
    for seed_idx, (seed, chunk_ids) in enumerate(remaining_jobs.items()):
        if wandb_parameters:
            # no need for chunks as it will run after merging
            wandb_parameters["samples_file"] = pipeline_utils.get_chunked_rs_filename(
                output_dir,
                random_seed=seed,
                chunk_id=None,
            )

        for chunk_id in chunk_ids:
            has_tasks = True

            # Configure server and client
            server_config, server_address, extra_arguments = pipeline_utils.configure_client(
                model=model,
                server_type=server_type,
                server_address=original_server_address,
                server_gpus=server_gpus,
                server_nodes=server_nodes,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
                server_container=server_container,
                extra_arguments=extra_arguments_original,
                get_random_port=get_random_port,
            )

            # Build generation command
            cmd = pipeline_utils.get_generation_cmd(
                input_file=input_file,
                input_dir=input_dir,
                random_seed=seed,
                output_dir=output_dir,
                extra_arguments=extra_arguments,
                eval_args=eval_args,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
                wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                script=generation_module,
            )

            # Create task group for each seed/chunk combination
            task_name = f"{expname}-rs{seed}" if seed is not None else expname
            if chunk_id is not None:
                task_name += f"-chunk{chunk_id}"

            # Handle dependent jobs (create multiple identical tasks)
            prev_task_groups = []
            for job_idx in range(dependent_jobs + 1):
                job_task_name = task_name
                if dependent_jobs > 0:
                    job_task_name += f"-job{job_idx}"

                # Create task group
                if server_config and server_config.get("num_gpus", 0) > 0:
                    # Use template for generation with server
                    task_group = PipelineTemplates.create_generation_with_server_pipeline(
                        name=job_task_name,
                        generation_cmd=pipeline_utils.wrap_python_path(cmd=cmd),
                        server_type=server_type,
                        model_path=model,
                        cluster_config=cluster_config,
                        server_gpus=server_gpus,
                        server_nodes=server_nodes,
                        with_sandbox=with_sandbox,
                        sandbox_port=None if get_random_port else 6000,
                    )
                else:
                    # Simple generation task without server
                    task_group = TaskGroup(job_task_name)
                    generation_task_def = TaskFactory.create_generation_task(
                        name=job_task_name,
                        cmd=pipeline_utils.wrap_python_path(cmd=cmd),
                        container=cluster_config["containers"]["nemo-skills"],
                        partition=partition,
                        installation_command=installation_command,
                    )
                    if exclusive:
                        generation_task_def.resources.slurm_kwargs = {"exclusive": True}
                    task_group.add_task(generation_task_def)

                # Add sandbox if requested
                if with_sandbox:
                    sandbox_task = TaskFactory.create_sandbox_task(
                        container=cluster_config["containers"]["sandbox"],
                        port=None if get_random_port else 6000,
                    )
                    task_group.add_task(sandbox_task)

                # Add dependencies
                if run_after:
                    task_group.add_dependency(run_after)

                # Add dependencies from previous jobs in the chain
                for prev_group in prev_task_groups:
                    task_group.add_dependency(prev_group)

                builder.add_task_group(task_group)
                all_task_groups.append(task_group)
                prev_task_groups = [task_group]

    # Build and run the experiment
    if has_tasks and not _reuse_exp:
        with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
            _ = builder.build_experiment(exp, log_dir)
            if not dry_run:
                pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)

        if _reuse_exp:
            return [group.get_task_id() for group in all_task_groups]
        else:
            return exp if has_tasks else None

    return None


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
