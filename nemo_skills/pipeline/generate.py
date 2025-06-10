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
from enum import Enum
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.inference.generate import GenerationTask
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import compute_chunk_ids, get_logger_name, setup_logging, str_ids_to_list

LOG = logging.getLogger(get_logger_name(__file__))


# TODO: move this away

# def get_genselect_cmd(
#     output_dir,
#     extra_arguments,
#     random_seed=None,
#     eval_args=None,
#     chunk_id=None,
#     num_chunks=None,
#     postprocess_cmd=None,
#     script: str = 'nemo_skills.inference.genselect',
#     output_prefix: str = "output",
# ):
#     if eval_args is not None:
#         raise ValueError("Cannot specify eval_args for genselect")
#     cmd = (
#         f"python -m {script} "
#         f"    ++skip_filled=True "
#         f"    ++input_dir={output_dir}/comparison_instances "
#         f"    ++output_dir={output_dir} "
#         f"    ++inference.random_seed={random_seed} "
#         f"    ++inference.temperature=0.7 "
#         f"    ++inference.tokens_to_generate=2048 "
#         f"    ++inference.top_k=0 "
#         f"    ++inference.top_p=0.95 "
#     )
#     cmd += f" {extra_arguments} "
#     return cmd, postprocess_cmd


class GenerationType(str, Enum):
    generate = "generate"
    reward = "reward"
    math_judge = "math_judge"
    # genselect = "genselect"


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
        ..., help="Path to the input data file. Can either specify input_file or input_dir, but not both. "
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
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    eval_args: str = typer.Option(
        None, help="Specify if need to run nemo_skills/evaluation/evaluate_results.py on the generation outputs"
    ),
    genselect_args: str = typer.Option(None, help="Can specify extra arguments to prepare the data for genselect"),
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
    output_prefix: str = typer.Option(
        "output", help="Optional base name for output .jsonl files. If provided, will be used in place of 'output'."
    ),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
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
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting generation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    chunking_enabled = (num_chunks is not None) or (chunk_ids is not None)
    if chunking_enabled and generation_type != GenerationType.generate:
        logging.error(
            "Chunking is enabled, but generation type is not 'generate'. "
            "Chunking is only supported for generation type 'generate'."
            "This may result in superfluous generation jobs."
        )
        raise ValueError("Chunking is only supported for generation type 'generate'")

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    if log_samples:
        wandb_parameters = {
            'name': wandb_name or expname,
            'project': wandb_project,
            'group': wandb_group,
        }
    else:
        wandb_parameters = None

    get_random_port = server_gpus != 8 and not exclusive

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

    get_server_command = server_command_factories[generation_type]
    get_cmd = client_command_factories[generation_type]
    cmd_script = client_command_scripts[generation_type]
    original_server_address = server_address

    # If GenerationType is `generate`, check if custom GenerationTask is provided via ctx.obj['generation_task_type']
    if (
        generation_type == GenerationType.generate
        and ctx.obj is not None
        and isinstance(ctx.obj, dict)
        and 'generation_task_type' in ctx.obj
    ):
        generation_task = ctx.obj['generation_task_type']  # type: type(GenerationTask)
        assert issubclass(
            generation_task, GenerationTask
        ), f"`generation_task_type` must be a subclass of GenerationTask"
        cmd_script = generation_task.get_generation_module()
        cmd_extra_args = generation_task.get_generation_default_args()
        cmd_script = f"{cmd_script.strip()} {cmd_extra_args.strip()}"

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
        output_prefix=output_prefix,
    )
    has_tasks = False

    with pipeline_utils.get_exp(expname, cluster_config) as exp:
        if generation_type == GenerationType.genselect:
            # Add the preprocessing command for genselect
            genselect_args = f" ++num_random_seeds={len(random_seeds)} ++output_dir={output_dir} " + (
                genselect_args if genselect_args is not None else ""
            )
            preprocess_cmd = f"python -m nemo_skills.inference.genselect_preprocess {genselect_args}"

            preprocess_task = pipeline_utils.add_task(
                exp,
                cmd=preprocess_cmd,
                task_name="preprocess_genselect",
                log_dir=f"{output_dir}/preprocess-logs",
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
            )
            initial_tasks = [preprocess_task]

        else:
            initial_tasks = None

        for seed_idx, (seed, chunk_ids) in enumerate(remaining_jobs.items()):
            if wandb_parameters:
                # no need for chunks as it will run after merging
                wandb_parameters['samples_file'] = pipeline_utils.get_chunked_rs_filename(
                    output_dir,
                    random_seed=seed,
                    chunk_id=None,
                    output_prefix=output_prefix,
                )
            for chunk_id in chunk_ids:
                has_tasks = True
                server_config, server_address, extra_arguments = pipeline_utils.configure_client(
                    model=model,
                    server_type=server_type,
                    server_address=original_server_address,
                    server_gpus=server_gpus,
                    server_nodes=server_nodes,
                    server_args=server_args,
                    server_entrypoint=server_entrypoint,
                    extra_arguments=extra_arguments_original,
                    get_random_port=get_random_port,
                )
                cmd = get_cmd(
                    random_seed=seed,
                    output_dir=output_dir,
                    extra_arguments=extra_arguments,
                    eval_args=eval_args,
                    chunk_id=chunk_id,
                    num_chunks=num_chunks,
                    output_prefix=output_prefix,
                    preprocess_cmd=preprocess_cmd,
                    postprocess_cmd=postprocess_cmd,
                    wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                    script=cmd_script,
                )
                prev_tasks = initial_tasks
                for _ in range(dependent_jobs + 1):
                    task_name = f'{expname}-rs{seed}' if seed is not None else expname
                    if chunk_id is not None:
                        task_name += f'-chunk{chunk_id}'
                    new_task = pipeline_utils.add_task(
                        exp,
                        cmd=pipeline_utils.wait_for_server(server_address=server_address, generation_commands=cmd),
                        task_name=task_name,
                        log_dir=log_dir,
                        container=cluster_config["containers"]["nemo-skills"],
                        cluster_config=cluster_config,
                        partition=partition,
                        time_min=time_min,
                        server_config=server_config,
                        with_sandbox=with_sandbox,
                        sandbox_port=None if get_random_port else 6000,
                        run_after=run_after,
                        reuse_code=reuse_code,
                        reuse_code_exp=reuse_code_exp,
                        task_dependencies=prev_tasks,
                        get_server_command=get_server_command,
                        slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                    )
                    prev_tasks = [new_task]
        if has_tasks:
            pipeline_utils.run_exp(exp, cluster_config)

    if has_tasks:
        return exp
    return None


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
