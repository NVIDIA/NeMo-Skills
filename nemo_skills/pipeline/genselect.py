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

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import get_logger_name, setup_logging, str_ids_to_list

LOG = logging.getLogger(get_logger_name(__file__))


def get_genselect_cmd(
    output_dir,
    extra_arguments,
    random_seed=None,
):
    cmd = (
        f"python -m nemo_skills.inference.genselect "
        f"    ++skip_filled=True "
        f"    ++input_dir={output_dir}/comparison_instances "
        f"    ++output_dir={output_dir} "
        f"    ++inference.random_seed={random_seed} "
        f"    ++inference.temperature=0.7 "
        f"    ++inference.tokens_to_generate=2048 "
        f"    ++inference.top_k=0 "
        f"    ++inference.top_p=0.95 "
    )
    cmd += f" {extra_arguments} "
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def genselect(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("generate", help="Nemo run experiment name"),
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
    preprocess_cmd: str = typer.Option(None, help="Command to run before generation"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after generation"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    preprocess_args: str = typer.Option(None, help="Can specify extra arguments to prepare the data for genselect"),
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
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
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
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting generation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive, server_type)

    if random_seeds and num_random_seeds:
        raise ValueError("Cannot specify both random_seeds and num_random_seeds")
    if num_random_seeds:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))
    if isinstance(random_seeds, str):
        random_seeds = str_ids_to_list(random_seeds)

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

    # Treat no random seeds as a single None seed to unify the code paths
    if not random_seeds:
        random_seeds = [None]

    remaining_jobs = pipeline_utils.get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir,
        random_seeds=random_seeds,
        chunk_ids=[None],
        rerun_done=rerun_done,
    )
    has_tasks = False
    extra_arguments_original = extra_arguments

    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        # Add the preprocessing command for genselect
        preprocess_args = f" ++num_random_seeds={len(random_seeds)} ++output_dir={output_dir} " + (
            preprocess_args if preprocess_args is not None else ""
        )
        task_preprocess_cmd = f"python -m nemo_skills.inference.genselect_preprocess {preprocess_args}"

        preprocess_task = pipeline_utils.add_task(
            exp,
            cmd=task_preprocess_cmd,
            task_name=f"{expname}-preprocess_genselect",
            log_dir=f"{output_dir}/preprocess-logs",
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            task_dependencies=_task_dependencies,
            run_after=run_after,
            reuse_code=reuse_code,
            reuse_code_exp=reuse_code_exp,
            slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            installation_command=installation_command,
        )
        for seed in remaining_jobs.keys():
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
            cmd = pipeline_utils.wrap_cmd(
                cmd=get_genselect_cmd(output_dir=output_dir, extra_arguments=extra_arguments, random_seed=seed),
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
            )
            prev_tasks = [preprocess_task]
            for _ in range(dependent_jobs + 1):
                task_name = f'{expname}-rs{seed}' if seed is not None else expname
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
                    slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                    installation_command=installation_command,
                )
                prev_tasks = [new_task]
        if has_tasks:
            pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)

    if _reuse_exp:
        return prev_tasks
    else:
        if has_tasks:
            return exp
        return None


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
