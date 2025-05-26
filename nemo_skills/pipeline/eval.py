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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import typer

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_mount_path,
    add_task,
    check_if_mounted,
    check_mounts,
    create_remote_directory,
    get_cluster_config,
    get_exp,
    get_free_port,
    get_generation_command,
    get_mounted_path,
    get_server_command,
    is_mounted_filepath,
    resolve_mount_paths,
    run_exp,
)
from nemo_skills.utils import compute_chunk_ids, get_chunked_filename, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


class ExtraDatasetType(str, Enum):
    copy = "copy"
    mount = "mount"


def get_greedy_cmd(
    benchmark,
    split,
    output_dir,
    output_name='output.jsonl',
    extra_eval_args="",
    extra_arguments="",
    extra_datasets=None,
    num_chunks=None,
    chunk_ids=None,
):
    benchmark_module, found_in_extra = get_dataset_module(benchmark, extra_datasets=extra_datasets)
    if found_in_extra:
        data_parameters = f"++input_file=/nemo_run/code/{Path(extra_datasets).name}/{benchmark}/{split}.jsonl"
    else:
        data_parameters = f"++dataset={benchmark} ++split={split}"

    extra_eval_args = f"{benchmark_module.DEFAULT_EVAL_ARGS} {extra_eval_args}"
    extra_arguments = f"{benchmark_module.DEFAULT_GENERATION_ARGS} {extra_arguments}"

    cmds = []
    if num_chunks is None or chunk_ids is None:
        chunk_params = ["++chunk_id=null ++num_chunks=null"]
        chunked_output_names = [output_name]
    else:
        chunk_params = [f"++chunk_id={chunk_id} ++num_chunks={num_chunks}" for chunk_id in chunk_ids]
        chunked_output_names = [get_chunked_filename(chunk_id, output_name) for chunk_id in chunk_ids]
    for chunk_param, chunked_output_name in zip(chunk_params, chunked_output_names):
        cmds.append(
            f'echo "Evaluating benchmark {benchmark}" && '
            f'python -m nemo_skills.inference.generate '
            f'    ++output_file={output_dir}/eval-results/{benchmark}/{output_name} '
            f'    {data_parameters} '
            f'    {chunk_param} '
            f'    {extra_arguments} && '
            f'python -m nemo_skills.evaluation.evaluate_results '
            f'    ++input_files={output_dir}/eval-results/{benchmark}/{chunked_output_name} {extra_eval_args}'
        )
    return cmds


def get_sampling_cmd(
    benchmark,
    split,
    output_dir,
    random_seed,
    extra_eval_args="",
    extra_arguments="",
    extra_datasets=None,
    num_chunks=None,
    chunk_ids=None,
):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark=benchmark,
        split=split,
        output_dir=output_dir,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
        extra_datasets=extra_datasets,
        num_chunks=num_chunks,
        chunk_ids=chunk_ids,
    )


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"
    sglang = "sglang"
    megatron = "megatron"


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
    benchmarks: str = typer.Option(
        ...,
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding. Has to be comma-separated "
        "if providing multiple benchmarks. E.g. gsm8k:4,human-eval:0",
    ),
    expname: str = typer.Option("eval", help="Name of the experiment"),
    model: str = typer.Option(None, help="Path to the model to be evaluated"),
    server_address: str = typer.Option(None, help="Address of the server hosting the model"),
    server_type: SupportedServers = typer.Option(help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    split: str = typer.Option('test', help="Data split to use for evaluation"),
    num_jobs: int = typer.Option(-1, help="Number of jobs to split the evaluation into"),
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
    add_greedy: bool = typer.Option(
        False,
        help="Whether to add greedy evaluation. Only applicable if num_samples > 0, otherwise greedy is default.",
    ),
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
    # extra_datasets_type: ExtraDatasetType = typer.Option(ExtraDatasetType.copy, help="How to handle extra datasets"),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
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

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if log_dir is None:
        log_dir = f"{output_dir}/eval-logs"

    output_dir, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    should_chunk_dataset = num_chunks is not None and chunk_ids is not None
    num_runs = len(chunk_ids) if should_chunk_dataset else 1

    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    get_random_port = server_gpus != 8 and not exclusive

    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_port = get_free_port(strategy="random") if get_random_port else 5000
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_entrypoint": server_entrypoint,
            "server_port": server_port,
        }
        # += is okay here because the args have already been copied in this context
        extra_arguments += f" ++server.server_type={server_type} "
        extra_arguments += f" ++server.host=localhost "
        extra_arguments += f" ++server.port={server_port} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
        )

    benchmarks = {k: int(v) for k, v in [b.split(":") for b in benchmarks.split(",")]}

    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
    # TODO(@titu1994): add support for extra_datasets_type in future pr
    # try:
    #     extra_datasets_type = extra_datasets_type.value
    # except AttributeError:
    #     pass

    # Check which benchmarks require sandbox
    benchmark_requires_sandbox = {}
    for benchmark in benchmarks.keys():
        benchmark_module, _ = get_dataset_module(benchmark, extra_datasets=extra_datasets)
        requires_sandbox = hasattr(benchmark_module, "DATASET_GROUP") and benchmark_module.DATASET_GROUP == "lean4"
        benchmark_requires_sandbox[benchmark] = requires_sandbox
        if requires_sandbox and not with_sandbox:
            LOG.warning("Found benchmark (%s) which requires sandbox mode, enabled sandbox for it.", benchmark)

    # Create evaluation commands as before
    eval_cmds = [
        (cmd, benchmark)
        for benchmark, rs_num in benchmarks.items()
        for cmd in get_greedy_cmd(
            benchmark,
            split,
            output_dir,
            extra_eval_args=extra_eval_args,
            extra_arguments=extra_arguments,
            extra_datasets=extra_datasets,
            num_chunks=num_chunks,
            chunk_ids=chunk_ids,
        )
        if add_greedy or rs_num == 0
    ]
    eval_cmds += [
        (cmd, benchmark)
        for benchmark, rs_num in benchmarks.items()
        for rs in range(starting_seed, starting_seed + rs_num)
        for cmd in get_sampling_cmd(
            benchmark,
            split,
            output_dir,
            rs,
            extra_eval_args=extra_eval_args,
            extra_arguments=extra_arguments,
            extra_datasets=extra_datasets,
            num_chunks=num_chunks,
            chunk_ids=chunk_ids,
        )
    ]
    if num_jobs == -1:
        num_jobs = len(eval_cmds)
    else:
        # TODO: should we keep num_jobs as the total max?
        num_jobs *= num_runs

    # Create job batches with benchmark info
    job_batches = []
    for i in range(num_jobs):
        cmds = []
        benchmarks_in_job = set()
        for cmd, benchmark in eval_cmds[i::num_jobs]:
            cmds.append(cmd)
            benchmarks_in_job.add(benchmark)
        job_batches.append((cmds, benchmarks_in_job))

    with get_exp(expname, cluster_config) as exp:
        for idx, (cmds, benchmarks_in_job) in enumerate(job_batches):
            # Check if any benchmark in this job requires sandbox
            job_needs_sandbox = with_sandbox or any(
                benchmark_requires_sandbox.get(b, False) for b in benchmarks_in_job
            )

            LOG.info("Launching task with command %s", " && ".join(cmds))
            add_task(
                exp,
                cmd=get_generation_command(server_address=server_address, generation_commands=" && ".join(cmds)),
                task_name=f'{expname}-{idx}',
                log_dir=log_dir,
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                server_config=server_config,
                with_sandbox=job_needs_sandbox,
                run_after=run_after,
                reuse_code_exp=reuse_code_exp,
                reuse_code=reuse_code,
                extra_package_dirs=[extra_datasets] if extra_datasets else None,
                get_server_command=get_server_command,
                sandbox_port=None if get_random_port else 6000,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
