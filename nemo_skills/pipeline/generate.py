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
import copy
import logging
from collections import defaultdict
from enum import Enum
from typing import List

import nemo_run as run
import typer

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import get_free_port, get_reward_server_command, get_server_command
from nemo_skills.utils import compute_chunk_ids, get_chunked_filename, setup_logging, str_ids_to_list

LOG = logging.getLogger(__file__)


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"
    sglang = "sglang"


def get_chunked_rs_filename(output_dir, random_seed=None, chunk_id=None):
    if random_seed is not None:
        output_file = f"{output_dir}/output-rs{random_seed}.jsonl"
    else:
        output_file = f"{output_dir}/output.jsonl"
    if chunk_id is not None:
        output_file = get_chunked_filename(chunk_id, output_file)
    return output_file


def get_cmd(
        output_dir, extra_arguments, random_seed=None, eval_args=None, chunk_id=None, num_chunks=None, postprocess_cmd=None, script: str = 'nemo_skills.inference.generate'
):
    # First get the unchunked filename for the output file
    output_file = get_chunked_rs_filename(f"{output_dir}", random_seed=random_seed)
    cmd = f"python -m {script} ++skip_filled=True ++output_file={output_file} "

    if random_seed is not None:
        cmd += (
            f"    ++inference.random_seed={random_seed} "
            f"    ++inference.temperature=1.0 "
            f"    ++inference.top_k=0 "
            f"    ++inference.top_p=0.95 "
        )

    if chunk_id is not None:
        cmd += f" ++num_chunks={num_chunks} ++chunk_id={chunk_id} "
        output_file = get_chunked_rs_filename(output_dir, random_seed=random_seed, chunk_id=chunk_id)
        donefiles = []
        # we are always waiting for all chunks in num_chunks, no matter chunk_ids in
        # the current run (as we don't want to merge partial jobs)
        for cur_chunk_id in range(num_chunks):
            donefile = f"{get_chunked_rs_filename(output_dir, random_seed=random_seed, chunk_id=cur_chunk_id)}.done"
            donefiles.append(donefile)

        if postprocess_cmd:
            postprocess_cmd += f" && touch {donefiles[chunk_id]} "
        else:
            postprocess_cmd = f"touch {donefiles[chunk_id]} "

        # getting file name as if there is no chunking since that's where we want to merge
        merged_output_file = get_chunked_rs_filename(output_dir, random_seed=random_seed)
        merge_cmd = (
            f"python -m nemo_skills.inference.merge_chunks {merged_output_file} "
            f"{' '.join([f[:-5] for f in donefiles])}"
        )
        postprocess_cmd += f" && {merge_cmd}"

    else:  # only writing a single status file
        if postprocess_cmd:
            postprocess_cmd += f" && touch {output_file}.done "
        else:
            postprocess_cmd = f"touch {output_file}.done "

    cmd += f" {extra_arguments} "
    if eval_args:
        cmd += (
            f" && python -m nemo_skills.evaluation.evaluate_results "
            f"    ++input_files={output_file} "
            f"    {eval_args} "
        )

    return cmd, postprocess_cmd


# TODO: support chunking for reward model and math judge


def get_rm_cmd(
    output_dir, extra_arguments, random_seed=None, eval_args=None, chunk_id=None, num_chunks=None, postprocess_cmd=None, script: str = 'nemo_skills.inference.reward_model'
):
    if eval_args is not None:
        raise ValueError("Cannot specify eval_args for reward model")

    cmd = (
        f"python -m {script} "
        f"    ++skip_filled=True "
        f"    ++output_dir={output_dir} "
        f"    ++random_seed={random_seed} "
    )
    cmd += f" {extra_arguments} "
    return cmd, postprocess_cmd


def get_math_judge_cmd(
    output_dir, extra_arguments, random_seed=None, eval_args=None, chunk_id=None, num_chunks=None, postprocess_cmd=None, script: str = 'nemo_skills.inference.llm_math_judge'
):
    if eval_args is not None:
        raise ValueError("Cannot specify eval_args for math judge")
    cmd = (
        f"python -m {script} "
        f"    ++skip_filled=True "
        f"    ++output_dir={output_dir} "
        f"    ++random_seed={random_seed} "
    )
    cmd += f" {extra_arguments} "
    return cmd, postprocess_cmd


def wrap_cmd(cmd, preprocess_cmd, postprocess_cmd, random_seed=None):
    if preprocess_cmd:
        if random_seed is not None:
            preprocess_cmd = preprocess_cmd.format(random_seed=random_seed)
        cmd = f" {preprocess_cmd} && {cmd} "
    if postprocess_cmd:
        if random_seed is not None:
            postprocess_cmd = postprocess_cmd.format(random_seed=random_seed)
        cmd = f" {cmd} && {postprocess_cmd} "
    return cmd


class GenerationType(str, Enum):
    generate = "generate"
    reward = "reward"
    math_judge = "math_judge"


server_command_factories = {
    GenerationType.generate: get_server_command,
    GenerationType.reward: get_reward_server_command,
    GenerationType.math_judge: get_server_command,
}

client_command_factories = {
    GenerationType.generate: get_cmd,
    GenerationType.reward: get_rm_cmd,
    GenerationType.math_judge: get_math_judge_cmd,
}

client_command_scripts = {
    GenerationType.generate: 'nemo_skills.inference.generate',
    GenerationType.reward: 'nemo_skills.inference.reward_model',
    GenerationType.math_judge: 'nemo_skills.inference.llm_math_judge',
}


def configure_client(
    generation_type,
    server_gpus,
    server_type,
    server_address,
    server_port,
    server_nodes,
    model,
    server_args,
    extra_arguments,
):
    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_port": server_port,
        }
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.host=localhost ++server.port={server_port} "
        )
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.base_url={server_address} ++server.model={model} "
        )
        server_port = None
    return server_config, extra_arguments, server_address, server_port


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def generate(
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
    generation_type: GenerationType = typer.Option(GenerationType.generate, help="Type of generation to perform"),
    server_type: SupportedServers = typer.Option(help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server"),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
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
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'

    chunking_enabled = (num_chunks is not None) or (chunk_ids is not None)
    if chunking_enabled and postprocess_cmd:
        logging.warning(
            "Chunking is enabled, but postprocess_cmd is also specified. "
            "Note that will be run for each chunk separately. Chunk merging "
            "will be performed after postprocess_cmd."
        )
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

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = f"{output_dir}/generation-logs"

    get_server_command = server_command_factories[generation_type]
    get_cmd = client_command_factories[generation_type]
    cmd_script = client_command_scripts[generation_type]
    original_server_address = server_address

    # If GenerationType is `generate`, check if custom GenerationTask is provided via ctx.obj['generation_task_type']
    if generation_type == GenerationType.generate and ctx.obj is not None and isinstance(ctx.obj, dict) and 'generation_task_type' in ctx.obj:
        generation_task = ctx.obj['generation_task_type']  # type: type(GenerationTask)
        assert issubclass(generation_task, GenerationTask), f"`generation_task_type` must be a subclass of GenerationTask"
        cmd_script = generation_task.get_generation_module()
        cmd_extra_args = generation_task.get_generation_default_args()
        cmd_script = f"{cmd_script.strip()} {cmd_extra_args.strip()}"

    with run.Experiment(expname) as exp:
        extra_arguments_original = extra_arguments
        # TODO: reduce code duplication
        if random_seeds:
            for chunk_id in chunk_ids:
                for seed in random_seeds:
                    server_port = get_free_port(strategy="random") if get_random_port else 5000
                    server_config, extra_arguments, server_address, server_port = configure_client(
                        generation_type=generation_type,
                        server_gpus=server_gpus,
                        server_type=server_type,
                        server_address=original_server_address,
                        server_port=server_port,
                        server_nodes=server_nodes,
                        model=model,
                        server_args=server_args,
                        extra_arguments=extra_arguments_original,
                    )
                    cmd, full_postprocess_cmd = get_cmd(
                        random_seed=seed,
                        output_dir=output_dir,
                        extra_arguments=extra_arguments,
                        eval_args=eval_args,
                        chunk_id=chunk_id,
                        num_chunks=num_chunks,
                        postprocess_cmd=postprocess_cmd,
                        script=cmd_script,
                    )
                    prev_tasks = None
                    for _ in range(dependent_jobs + 1):
                        new_task = add_task(
                            exp,
                            cmd=wrap_cmd(
                                get_generation_command(server_address=server_address, generation_commands=cmd),
                                preprocess_cmd,
                                full_postprocess_cmd,
                                random_seed=seed,
                            ),
                            task_name=f'{expname}-rs{seed}',
                            log_dir=log_dir,
                            container=cluster_config["containers"]["nemo-skills"],
                            cluster_config=cluster_config,
                            partition=partition,
                            time_min=time_min,
                            server_config=server_config,
                            with_sandbox=True,
                            sandbox_port=None if get_random_port else 6000,
                            run_after=run_after,
                            reuse_code=reuse_code,
                            reuse_code_exp=reuse_code_exp,
                            task_dependencies=prev_tasks,
                            get_server_command=get_server_command,
                            slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                        )
                        prev_tasks = [new_task]
        else:
            for chunk_id in chunk_ids:
                server_port = get_free_port(strategy="random") if get_random_port else 5000
                server_config, extra_arguments, server_address, server_port = configure_client(
                    generation_type=generation_type,
                    server_gpus=server_gpus,
                    server_type=server_type,
                    server_address=original_server_address,
                    server_port=server_port,
                    server_nodes=server_nodes,
                    model=model,
                    server_args=server_args,
                    extra_arguments=extra_arguments_original,
                )

                cmd, full_postprocess_cmd = get_cmd(
                    random_seed=None,
                    output_dir=output_dir,
                    extra_arguments=extra_arguments,
                    eval_args=eval_args,
                    chunk_id=chunk_id,
                    num_chunks=num_chunks,
                    postprocess_cmd=postprocess_cmd,
                    script=cmd_script,
                )
                prev_tasks = None

                for _ in range(dependent_jobs + 1):
                    new_task = add_task(
                        exp,
                        cmd=wrap_cmd(
                            get_generation_command(server_address=server_address, generation_commands=cmd),
                            preprocess_cmd,
                            full_postprocess_cmd,
                        ),
                        task_name=expname,
                        log_dir=log_dir,
                        container=cluster_config["containers"]["nemo-skills"],
                        cluster_config=cluster_config,
                        partition=partition,
                        time_min=time_min,
                        server_config=server_config,
                        with_sandbox=True,
                        sandbox_port=None if get_random_port else 6000,
                        run_after=run_after,
                        reuse_code=reuse_code,
                        reuse_code_exp=reuse_code_exp,
                        task_dependencies=prev_tasks,
                        get_server_command=get_server_command,
                        slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                    )
                    prev_tasks = [new_task]
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
