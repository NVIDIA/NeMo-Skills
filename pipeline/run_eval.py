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

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, WRAPPER_HELP, get_server_command, launch_job

try:
    from nemo_skills.inference.generate_solutions import HELP_MESSAGE
except ImportError:
    HELP_MESSAGE = """
To see all supported agruments, nemo_skills package needs to be installed.
Please note that it is not recommended to install Python packages on a slurm cluster login node.
"""
from nemo_skills.utils import setup_logging

SCRIPT_HELP = """
This script can be used to run evaluation of a model on a set of benchmarks.
It can run both greedy decoding and sampling and can parallelize generations
across multiple nodes. It uses nemo_skills/inference/generate_solutions.py
to generate solutions and nemo_skills/evaluation/evaluate_results.py to
evaluate them. It will set reasonable defaults for most of the generation parameters,
but you can override any of them by directly providing corresponding arguments
in the Hydra format.
"""


def get_greedy_cmd(benchmark, output_name='output-greedy.jsonl', extra_eval_args="", extra_arguments=""):
    return f"""echo "Evaluating benchmark {benchmark}" && \
python nemo_skills/inference/generate_solutions.py \
    server.server_type={{server_type}} \
    prompt.context_type=empty \
    +dataset={benchmark} \
    output_file=/results/{benchmark}/{output_name} \
    {extra_arguments} && \
python nemo_skills/evaluation/evaluate_results.py \
    prediction_jsonl_files=/results/{benchmark}/{output_name} {extra_eval_args} && \
"""


def get_sampling_cmd(benchmark, random_seed, extra_eval_args="", extra_arguments=""):
    extra_arguments = f" inference.random_seed={random_seed} inference.temperature=0.7 {extra_arguments}"
    return get_greedy_cmd(
        benchmark,
        output_name=f"output-rs{random_seed}.jsonl",
        extra_eval_args=extra_eval_args,
        extra_arguments=extra_arguments,
    )


# default number of samples for majority voting
BENCHMARKS = {
    "gsm8k": 8,
    "math": 4,
    "tabmwp": 4,
}


SLURM_CMD = """
nvidia-smi && \
cd /code && \
export PYTHONPATH=$PYTHONPATH:/code && \
export HF_TOKEN={HF_TOKEN} && \
if [ $SLURM_PROCID -eq 0 ]; then \
    {{ {server_start_cmd} 2>&1 | tee /tmp/server_logs.txt & }} && sleep 1 && \
    echo "Waiting for the server to start" && \
    tail -n0 -f /tmp/server_logs.txt | sed '/{server_wait_string}/ q' && \
    {eval_cmds} \
    pkill -f nemo_skills/inference/server; \
else \
    {server_start_cmd}; \
fi \
"""


MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model,{output_dir}:/results"
JOB_NAME = "eval-{model_name}"

if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser(usage=WRAPPER_HELP + '\n\n' + SCRIPT_HELP + '\n\nscript arguments:\n\n' + HELP_MESSAGE)
    wrapper_args = parser.add_argument_group('wrapper arguments')
    wrapper_args.add_argument("--model_path", required=True)
    wrapper_args.add_argument("--server_type", choices=('nemo', 'tensorrt_llm', 'vllm'), default='tensorrt_llm')
    wrapper_args.add_argument("--output_dir", required=True)
    wrapper_args.add_argument("--num_gpus", type=int, required=True)
    wrapper_args.add_argument("--starting_seed", type=int, default=0)
    wrapper_args.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        help="Need to be in a format <benchmark>:<num samples for majority voting>. "
        "Use <benchmark>:0 to only run greedy decoding.",
    )
    wrapper_args.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    wrapper_args.add_argument(
        "--num_jobs",
        type=int,
        default=-1,
        help="Will launch this many separate jobs and split the benchmarks across them. "
        "Set -1 to run each benchmark / random seed as a separate job.",
    )
    wrapper_args.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    wrapper_args.add_argument(
        "--extra_eval_args",
        default="",
        help="Any extra arguments to pass to nemo_skills/evaluation/evaluate_results.py",
    )

    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    args.model_path = Path(args.model_path).absolute()
    args.output_dir = Path(args.output_dir).absolute()

    server_start_cmd, num_tasks, server_wait_string = get_server_command(
        args.server_type, args.num_gpus, args.num_nodes, args.model_path.name
    )

    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "output_dir": args.output_dir,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "server_type": args.server_type,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # needed for some of the models, so making an option to pass it in
        "server_wait_string": server_wait_string,
    }

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # if benchmarks are specified, only run those
    if args.benchmarks:
        BENCHMARKS = {k: int(v) for k, v in [b.split(":") for b in args.benchmarks]}

    eval_cmds = [
        get_greedy_cmd(benchmark, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments)
        for benchmark in BENCHMARKS.keys()
    ]
    eval_cmds += [
        get_sampling_cmd(benchmark, rs, extra_eval_args=args.extra_eval_args, extra_arguments=extra_arguments)
        for benchmark, rs_num in BENCHMARKS.items()
        for rs in range(args.starting_seed, args.starting_seed + rs_num)
    ]
    if args.num_jobs == -1:
        args.num_jobs = len(eval_cmds)

    # splitting eval cmds equally across num_jobs nodes
    eval_cmds = [" ".join(eval_cmds[i :: args.num_jobs]) for i in range(args.num_jobs)]

    for idx, eval_cmd in enumerate(eval_cmds):
        extra_sbatch_args = ["--parsable", f"--output={args.output_dir}/slurm_logs_eval{idx}.log"]
        launch_job(
            cmd=SLURM_CMD.format(**format_dict, eval_cmds=eval_cmd.format(**format_dict)),
            num_nodes=args.num_nodes,
            tasks_per_node=num_tasks,
            gpus_per_node=format_dict["num_gpus"],
            job_name=JOB_NAME.format(**format_dict),
            container=CLUSTER_CONFIG["containers"][args.server_type],
            mounts=MOUNTS.format(**format_dict),
            partition=args.partition,
            with_sandbox=True,
            extra_sbatch_args=extra_sbatch_args,
        )
