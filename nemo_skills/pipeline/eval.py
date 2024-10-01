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
from enum import Enum

import nemo_run as run
import typer

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, get_generation_command, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_greedy_cmd(
    benchmark, split, output_dir, output_name='output-greedy.jsonl', extra_eval_args="", extra_arguments=""
):
    benchmark_module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")

    extra_eval_args = f"{benchmark_module.DEFAULT_EVAL_ARGS} {extra_eval_args}"
    extra_arguments = f"{benchmark_module.DEFAULT_GENERATION_ARGS} {extra_arguments}"
    cmd = (
        f'echo "Evaluating benchmark {benchmark}" && '
        f'python -m nemo_skills.inference.generate '
        f'    ++dataset={benchmark} '
        f'    ++split={split} '
        f'    ++output_file={output_dir}/eval-results/{benchmark}/{output_name} '
        f'    {extra_arguments} && '
        f'python -m nemo_skills.evaluation.evaluate_results '
        f'    ++input_files={output_dir}/eval-results/{benchmark}/{output_name} {extra_eval_args}'
    )
    return cmd


def get_sampling_cmd(benchmark, split, output_dir, random_seed, extra_eval_args="", extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark=benchmark,
        split=split,
        output_dir=output_dir,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
    )


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    nemo = "nemo"
    openai = "openai"


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def eval(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
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
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    split: str = typer.Option('test', help="Data split to use for evaluation"),
    num_jobs: int = typer.Option(-1, help="Number of jobs to split the evaluation into"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    extra_eval_args: str = typer.Option("", help="Additional arguments for evaluation"),
    skip_greedy: bool = typer.Option(False, help="Whether to skip greedy evaluation"),
    run_after: str = typer.Option(None, help="Task to run after the evaluation"),
):
    """Evaluate a model on specified benchmarks.

    Any extra arguments will be directly passed to nemo_skills.inference.generate
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting evaluation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)

    if server_address is None:  # we need to host the model
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = "localhost:5000"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
        }
        extra_arguments += f" ++server.server_type={server_type} "
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments += (
            f" ++server.server_type={server_type} ++server.base_url={server_address} ++server.model={model} "
        )

    benchmarks = {k: int(v) for k, v in [b.split(":") for b in benchmarks.split(",")]}

    eval_cmds = (
        [
            get_greedy_cmd(
                benchmark,
                split,
                output_dir,
                extra_eval_args=extra_eval_args,
                extra_arguments=extra_arguments,
            )
            for benchmark in benchmarks.keys()
        ]
        if not skip_greedy
        else []
    )
    eval_cmds += [
        get_sampling_cmd(
            benchmark,
            split,
            output_dir,
            rs,
            extra_eval_args=extra_eval_args,
            extra_arguments=extra_arguments,
        )
        for benchmark, rs_num in benchmarks.items()
        for rs in range(starting_seed, starting_seed + rs_num)
    ]
    if num_jobs == -1:
        num_jobs = len(eval_cmds)

    # splitting eval cmds equally across num_jobs nodes
    eval_cmds = [" && ".join(eval_cmds[i::num_jobs]) for i in range(num_jobs)]

    with run.Experiment(expname) as exp:
        for idx, eval_cmd in enumerate(eval_cmds):
            LOG.info("Launching task with command %s", eval_cmd)
            add_task(
                exp,
                cmd=get_generation_command(server_address=server_address, generation_commands=eval_cmd),
                task_name=f'eval-{idx}',
                log_dir=f"{output_dir}/eval-logs",
                container=cluster_config["containers"]["nemo-skills"],
                cluster_config=cluster_config,
                partition=partition,
                server_config=server_config,
                with_sandbox=True,
                run_after=run_after,
            )
        run_exp(exp, cluster_config)


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
