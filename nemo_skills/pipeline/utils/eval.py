# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import ExtraDatasetType, get_dataset_module
from nemo_skills.inference.generate import GenerationTask
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import compute_chunk_ids, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def add_default_args(cluster_config, benchmark, split, data_dir, extra_datasets_type, extra_datasets):
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

    prompt_config_arg = f"++prompt_config={benchmark_module.PROMPT_CONFIG}"
    benchmark_gen_args = f"{prompt_config_arg} {benchmark_module.GENERATION_ARGS}"
    requires_sandbox = getattr(benchmark_module, "REQUIRES_SANDBOX", False)

    generation_module = getattr(benchmark_module, "GENERATION_MODULE", "nemo_skills.inference.generate")

    return input_file, benchmark_gen_args, benchmark_module.EVAL_ARGS, requires_sandbox, generation_module


def prepare_eval_commands(
    cluster_config,
    benchmarks,
    split,
    extra_datasets,
    num_jobs,
    starting_seed,
    output_dir,
    num_chunks,
    chunk_ids,
    rerun_done,
    model,
    server_type,
    server_address,
    server_gpus,
    server_nodes,
    server_args,
    server_entrypoint,
    extra_arguments,
    data_dir,
    extra_datasets_type,
    exclusive,
    with_sandbox,
    wandb_parameters,
    extra_eval_args,
):
    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    benchmarks = {k: int(v) for k, v in [b.split(":") if ":" in b else (b, 0) for b in benchmarks.split(",")]}
    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")

    if num_jobs is None:
        if cluster_config['executor'] == 'slurm':
            num_jobs = -1  # -1 means run all benchmarks in parallel
        else:
            # for local executor, it makes no sense to use other values
            num_jobs = 1

    benchmark_remaining_jobs = {}
    total_evals = 0
    for benchmark, rs_num in benchmarks.items():
        if rs_num == 0:
            random_seeds = [None]
        else:
            random_seeds = list(range(starting_seed, starting_seed + rs_num))

        benchmark_output_dir = f"{output_dir}/eval-results/{benchmark}"
        benchmark_remaining_jobs[benchmark] = pipeline_utils.get_remaining_jobs(
            cluster_config=cluster_config,
            output_dir=benchmark_output_dir,
            random_seeds=random_seeds,
            chunk_ids=chunk_ids,
            rerun_done=rerun_done,
        )
        for seed_idx, (seed, benchmark_chunk_ids) in enumerate(benchmark_remaining_jobs[benchmark].items()):
            total_evals += len(benchmark_chunk_ids)

    if num_jobs < 0:
        # if num_jobs is -1, we run all benchmarks in parallel
        num_jobs = total_evals

    if num_jobs == 0:
        return None

    evals_per_job = total_evals // num_jobs if num_jobs > 0 else total_evals
    remainder = total_evals % num_jobs
    eval_to_job_map = []
    for i in range(num_jobs):
        count = evals_per_job + (1 if i < remainder else 0)
        eval_to_job_map.extend([i] * count)

    cur_job_idx = 0
    get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive, server_type)
    job_server_config, job_server_address, job_extra_arguments = pipeline_utils.configure_client(
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

    cur_eval = 0
    job_batches = []
    job_cmds = []
    job_benchmarks = set()
    has_tasks = False

    benchmark_required_sandbox = {}

    for benchmark, rs_num in benchmarks.items():
        bench_input_file, bench_gen_args, bench_eval_args, requires_sandbox, generation_module = add_default_args(
            cluster_config,
            benchmark,
            split,
            data_dir,
            extra_datasets_type,
            extra_datasets,
        )
        benchmark_required_sandbox[benchmark] = requires_sandbox
        if requires_sandbox and not with_sandbox:
            LOG.warning("Found benchmark (%s) which requires sandbox, enabled sandbox for it.", benchmark)

        if rs_num == 0:
            random_seeds = [None]
        else:
            random_seeds = list(range(starting_seed, starting_seed + rs_num))

        benchmark_output_dir = f"{output_dir}/eval-results/{benchmark}"
        for seed_idx, (seed, benchmark_chunk_ids) in enumerate(benchmark_remaining_jobs[benchmark].items()):
            if wandb_parameters:
                # no need for chunks as it will run after merging
                wandb_parameters['samples_file'] = pipeline_utils.get_chunked_rs_filename(
                    benchmark_output_dir,
                    random_seed=seed,
                    chunk_id=None,
                )
            for chunk_id in benchmark_chunk_ids:
                has_tasks = True
                job_benchmarks.add(benchmark)

                generation_task = importlib.import_module(generation_module)
                if not hasattr(generation_task, 'GENERATION_TASK_CLASS'):
                    raise ValueError(
                        f"Module {generation_module} does not have a GENERATION_TASK_CLASS attribute. "
                        "Please provide a valid generation module."
                    )
                generation_task = generation_task.GENERATION_TASK_CLASS
                if (
                    generation_task.get_server_command_fn.__func__ != GenerationTask.get_server_command_fn.__func__
                    and num_jobs != total_evals
                ):
                    raise ValueError(
                        f"Class {generation_task} overrides get_server_command_fn, "
                        "which is not supported for evaluation when grouping jobs."
                    )

                cmd = pipeline_utils.get_generation_cmd(
                    input_file=bench_input_file,
                    output_dir=benchmark_output_dir,
                    extra_arguments=f"{generation_task.get_generation_default_args()} {bench_gen_args} {job_extra_arguments}",
                    random_seed=seed,
                    eval_args=f"{bench_eval_args} {extra_eval_args}",
                    chunk_id=chunk_id,
                    num_chunks=num_chunks,
                    script=generation_module,
                    # only logging for the first seed
                    wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                )
                job_cmds.append(cmd)

                if cur_eval == total_evals - 1 or cur_job_idx != eval_to_job_map[cur_eval + 1]:
                    job_needs_sandbox = any(benchmark_required_sandbox[b] for b in job_benchmarks)
                    job_batches.append(
                        (
                            job_cmds,
                            job_needs_sandbox,
                            job_server_config,
                            job_server_address,
                            # a check above guarantees that this is the same for all tasks in a job
                            generation_task.get_server_command_fn(),
                        )
                    )
                    job_server_config, job_server_address, job_extra_arguments = pipeline_utils.configure_client(
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
                    cur_job_idx += 1
                    job_cmds = []
                    job_benchmarks = set()

                cur_eval += 1
