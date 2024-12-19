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

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import get_free_port, get_reward_server_command, get_server_command
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"


def get_cmd(output_dir, extra_arguments, random_seed=None, eval_args=None):
    if random_seed is not None:
        output_file = f"{output_dir}/generation/output-rs{random_seed}.jsonl"
    else:
        output_file = f"{output_dir}/generation/output.jsonl"
    cmd = f"python -m nemo_skills.inference.generate ++skip_filled=True ++output_file={output_file} "
    if random_seed is not None:
        cmd += (
            f"    ++inference.random_seed={random_seed} "
            f"    ++inference.temperature=1.0 "
            f"    ++inference.top_k=0 "
            f"    ++inference.top_p=0.95 "
        )
    cmd += f" {extra_arguments} "
    if eval_args:
        cmd += (
            f" && python -m nemo_skills.evaluation.evaluate_results "
            f"    ++input_files={output_file} "
            f"    {eval_args} "
        )
    return cmd


def get_rm_cmd(output_dir, extra_arguments, random_seed=None, eval_args=None):
    if eval_args is not None:
        raise ValueError("Cannot specify eval_args for reward model")
    cmd = (
        f"python -m nemo_skills.inference.reward_model "
        f"    ++skip_filled=True "
        f"    ++output_dir={output_dir} "
        f"    ++random_seed={random_seed} "
    )
    cmd += f" {extra_arguments} "
    return cmd


def get_math_judge_cmd(output_dir, extra_arguments, random_seed=None, eval_args=None):
    if eval_args is not None:
        raise ValueError("Cannot specify eval_args for math judge")
    cmd = (
        f"python -m nemo_skills.inference.llm_math_judge "
        f"    ++skip_filled=True "
        f"    ++output_dir={output_dir} "
        f"    ++random_seed={random_seed} "
    )
    cmd += f" {extra_arguments} "
    return cmd


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
        server_port = get_free_port()
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
        extra_arguments += f" ++server.server_type={server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
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
    random_seeds: List[int] = typer.Option(None, help="List of random seeds to use for generation"),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
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
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    if random_seeds and num_random_seeds:
        raise ValueError("Cannot specify both random_seeds and num_random_seeds")
    if num_random_seeds:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = f"{output_dir}/generation-logs"

    get_server_command = server_command_factories[generation_type]
    get_cmd = client_command_factories[generation_type]
    original_server_address = server_address

    with run.Experiment(expname) as exp:
        if random_seeds:
            for seed in random_seeds:
                server_port = get_free_port()
                server_config, extra_arguments, server_address, server_port = configure_client(
                    generation_type=generation_type,
                    server_gpus=server_gpus,
                    server_type=server_type,
                    server_address=original_server_address,
                    server_port=server_port,
                    server_nodes=server_nodes,
                    model=model,
                    server_args=server_args,
                    extra_arguments=extra_arguments,
                )

                cmd = get_cmd(
                    random_seed=seed,
                    output_dir=output_dir,
                    extra_arguments=extra_arguments,
                    eval_args=eval_args,
                )
                prev_tasks = None

                for _ in range(dependent_jobs + 1):
                    new_task = add_task(
                        exp,
                        cmd=wrap_cmd(
                            get_generation_command(server_address=server_address, generation_commands=cmd),
                            preprocess_cmd,
                            postprocess_cmd,
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
                        run_after=run_after,
                        reuse_code_exp=reuse_code_exp,
                        task_dependencies=prev_tasks,
                        get_server_command=get_server_command,
                    )
                    prev_tasks = [new_task]
        else:
            server_port = get_free_port()
            server_config, extra_arguments, server_address, server_port = configure_client(
                generation_type=generation_type,
                server_gpus=server_gpus,
                server_type=server_type,
                server_address=original_server_address,
                server_port=server_port,
                server_nodes=server_nodes,
                model=model,
                server_args=server_args,
                extra_arguments=extra_arguments,
            )
            cmd = get_cmd(
                random_seed=None,
                output_dir=output_dir,
                extra_arguments=extra_arguments,
                eval_args=eval_args,
            )
            prev_tasks = None
            for _ in range(dependent_jobs + 1):
                new_task = add_task(
                    exp,
                    cmd=wrap_cmd(
                        get_generation_command(server_address=server_address, generation_commands=cmd),
                        preprocess_cmd,
                        postprocess_cmd,
                    ),
                    task_name=expname,
                    log_dir=log_dir,
                    container=cluster_config["containers"]["nemo-skills"],
                    cluster_config=cluster_config,
                    partition=partition,
                    time_min=time_min,
                    server_config=server_config,
                    with_sandbox=True,
                    run_after=run_after,
                    reuse_code_exp=reuse_code_exp,
                    task_dependencies=prev_tasks,
                    get_server_command=get_server_command,
                )
                prev_tasks = [new_task]
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
