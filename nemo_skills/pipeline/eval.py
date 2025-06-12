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
import os
from pathlib import Path
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import ExtraDatasetType, get_dataset_module
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import compute_chunk_ids, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def add_default_args(
    cluster_config, benchmark, split, data_dir, extra_eval_args, extra_arguments, extra_datasets_type, extra_datasets
):
    benchmark_module, data_path, is_on_cluster = get_dataset_module(
        dataset=benchmark,
        data_dir=data_dir,
        cluster_config=cluster_config,
        extra_datasets=extra_datasets,
        extra_datasets_type=extra_datasets_type,
    )
    benchmark = benchmark.replace('.', '/')

    if split is None:
        split = getattr(benchmark_module, "EVAL_SPLIT", "test")
    if not is_on_cluster:
        if pipeline_utils.is_mounted_filepath(cluster_config, data_path):
            input_file = f"{data_path}/{benchmark}/{split}.jsonl"
            unmounted_input_file = pipeline_utils.get_unmounted_path(cluster_config, input_file)
            unmounted_path = str(Path(__file__).parents[2] / unmounted_input_file.replace('/nemo_run/code/', ''))
        else:
            # will be copied over in this case as it must come from extra datasets
            input_file = f"/nemo_run/code/{Path(data_path).name}/{benchmark}/{split}.jsonl"
            unmounted_path = Path(data_path) / benchmark / f"{split}.jsonl"
    else:
        # on cluster we will always use the mounted path
        input_file = f"{data_path}/{benchmark}/{split}.jsonl"
        unmounted_path = pipeline_utils.get_unmounted_path(cluster_config, input_file)

    unmounted_path = str(unmounted_path)
    # checking if data file exists (can check locally as well)
    if is_on_cluster:
        if not pipeline_utils.cluster_path_exists(cluster_config, unmounted_path):
            raise ValueError(
                f"Data file {unmounted_path} does not exist on cluster. "
                "Please check the benchmark and split parameters. "
                "Did you forget to run prepare data commands?"
            )
    else:
        if not Path(unmounted_path).exists():
            raise ValueError(
                f"Data file {unmounted_path} does not exist locally. "
                "Please check the benchmark and split parameters. "
                "Did you forget to run prepare data commands?"
            )

    extra_eval_args = f"{benchmark_module.EVAL_ARGS} {extra_eval_args}"
    prompt_config_arg = f"++prompt_config={benchmark_module.PROMPT_CONFIG}"
    default_arguments = f"{prompt_config_arg} {benchmark_module.GENERATION_ARGS}"
    extra_arguments = f"{default_arguments} {extra_arguments}"

    requires_sandbox = hasattr(benchmark_module, "DATASET_GROUP") and benchmark_module.DATASET_GROUP == "lean4"

    return input_file, extra_arguments, extra_eval_args, requires_sandbox


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def eval(
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
    benchmarks: str = typer.Option(
        ...,
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding. Has to be comma-separated "
        "if providing multiple benchmarks. E.g. gsm8k:4,human-eval:0",
    ),
    expname: str = typer.Option("eval", help="Name of the experiment"),
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
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    split: str = typer.Option(
        None,
        help="Data split to use for evaluation. Will use benchmark-specific default or 'test' if it's not defined.",
    ),
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
    extra_eval_args: str = typer.Option("", help="Additional arguments for evaluation"),
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
    """Evaluate a model on specified benchmarks.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting evaluation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        server_type = server_type.value
    except AttributeError:
        pass
    try:
        extra_datasets_type = extra_datasets_type.value
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

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    env_vars = pipeline_utils.get_env_variables(cluster_config)
    data_dir = data_dir or env_vars.get("NEMO_SKILLS_DATA_DIR") or os.environ.get("NEMO_SKILLS_DATA_DIR")

    if extra_datasets_type == ExtraDatasetType.cluster and cluster_config['executor'] != 'slurm':
        raise ValueError(
            "Extra datasets type is set to 'cluster', but the executor is not 'slurm'. "
            "Please use 'local' or change the cluster config."
        )

    if log_dir is None:
        log_dir = f"{output_dir}/eval-logs"

    output_dir, data_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None, data_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    # TODO: random port will not really work here as we need to move this in the loop
    get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive, server_type)
    server_config, server_address, extra_arguments = pipeline_utils.configure_client(
        model=model,
        server_type=server_type,
        server_address=server_address,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_args=server_args,
        server_entrypoint=server_entrypoint,
        extra_arguments=extra_arguments,
        get_random_port=get_random_port,
    )

    benchmarks = {k: int(v) for k, v in [b.split(":") for b in benchmarks.split(",")]}
    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")

    eval_cmds = []
    benchmark_requires_sandbox = {}
    has_tasks = False

    for benchmark, rs_num in benchmarks.items():
        bench_input_file, bench_gen_args, bench_eval_args, requires_sandbox = add_default_args(
            cluster_config,
            benchmark,
            split,
            data_dir,
            extra_eval_args,
            extra_arguments,
            extra_datasets_type,
            extra_datasets,
        )
        benchmark_requires_sandbox[benchmark] = requires_sandbox
        if requires_sandbox and not with_sandbox:
            LOG.warning("Found benchmark (%s) which requires sandbox mode, enabled sandbox for it.", benchmark)

        if rs_num == 0:
            random_seeds = [None]
        else:
            random_seeds = list(range(starting_seed, starting_seed + rs_num))

        benchmark_output_dir = f"{output_dir}/eval-results/{benchmark}"
        remaining_jobs = pipeline_utils.get_remaining_jobs(
            cluster_config=cluster_config,
            output_dir=benchmark_output_dir,
            random_seeds=random_seeds,
            chunk_ids=chunk_ids,
            rerun_done=rerun_done,
        )
        for seed_idx, (seed, benchmark_chunk_ids) in enumerate(remaining_jobs.items()):
            if wandb_parameters:
                # no need for chunks as it will run after merging
                wandb_parameters['samples_file'] = pipeline_utils.get_chunked_rs_filename(
                    benchmark_output_dir,
                    random_seed=seed,
                    chunk_id=None,
                )
            for chunk_id in benchmark_chunk_ids:
                has_tasks = True
                cmd = pipeline_utils.get_generation_cmd(
                    input_file=bench_input_file,
                    output_dir=benchmark_output_dir,
                    extra_arguments=bench_gen_args,
                    random_seed=seed,
                    eval_args=bench_eval_args,
                    chunk_id=chunk_id,
                    num_chunks=num_chunks,
                    # only logging for the first seed
                    wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                )
                eval_cmds.append((cmd, benchmark))

    if num_jobs is None:
        if cluster_config['executor'] == 'slurm':
            num_jobs = len(eval_cmds)
        else:
            # for local executor, it makes no sense to use other values
            num_jobs = 1
    if num_jobs < 0:
        num_jobs = len(eval_cmds)

    # Create job batches with benchmark info
    job_batches = []
    for i in range(num_jobs):
        cmds = []
        benchmarks_in_job = set()
        for cmd, benchmark in eval_cmds[i::num_jobs]:
            cmds.append(cmd)
            benchmarks_in_job.add(benchmark)
        job_batches.append((cmds, benchmarks_in_job))

    should_package_extra_datasets = extra_datasets and extra_datasets_type == ExtraDatasetType.local
    with pipeline_utils.get_exp(expname, cluster_config) as exp:
        for idx, (cmds, benchmarks_in_job) in enumerate(job_batches):
            # Check if any benchmark in this job requires sandbox
            job_needs_sandbox = with_sandbox or any(
                benchmark_requires_sandbox.get(b, False) for b in benchmarks_in_job
            )

            prev_tasks = None
            for _ in range(dependent_jobs + 1):
                new_task = pipeline_utils.add_task(
                    exp,
                    cmd=pipeline_utils.wait_for_server(
                        server_address=server_address, generation_commands=" && ".join(cmds)
                    ),
                    task_name=f'{expname}-{idx}',
                    log_dir=log_dir,
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    partition=partition,
                    time_min=time_min,
                    server_config=server_config,
                    with_sandbox=job_needs_sandbox,
                    sandbox_port=None if get_random_port else 6000,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    reuse_code=reuse_code,
                    task_dependencies=prev_tasks,
                    extra_package_dirs=[extra_datasets] if should_package_extra_datasets else None,
                    slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                )
                prev_tasks = [new_task]
        if has_tasks:
            pipeline_utils.run_exp(exp, cluster_config)

    if has_tasks:
        return exp
    return None


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
