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

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, get_server_command, launch_job

from nemo_skills.utils import setup_logging

SLURM_CMD = """
nvidia-smi && \
cd /code && \
export PYTHONPATH=$PYTHONPATH:/code && \
export HF_TOKEN={HF_TOKEN} && \
if [ $SLURM_PROCID -eq 0 ]; then \
    {{ {server_start_cmd} 2>&1 | tee /tmp/server_logs.txt & }} && sleep 1 && \
    echo "Waiting for the server to start" && \
    tail -n0 -f /tmp/server_logs.txt | sed '/{server_wait_string}/ q' && \
    python nemo_skills/inference/generate_solutions.py \
        server.server_type={server_type} \
        skip_filled=True \
        inference.random_seed={random_seed} \
        inference.temperature=1.0 \
        inference.top_k=0 \
        inference.top_p=0.95 \
        output_file=/results/output-rs{random_seed}.jsonl \
        {extra_arguments} && \
    python nemo_skills/evaluation/evaluate_results.py \
        prediction_jsonl_files=/results/output-rs{random_seed}.jsonl {extra_eval_args} && \
    pkill -f nemo_skills/inference/server; \
else \
    {server_start_cmd}; \
fi \
"""


# TODO: when parameters are incorrect, the error is displayed in a bizarre way


MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model,{output_dir}:/results"
LOGS = "{output_dir}/slurm_logs-rs{random_seed}.txt"
JOB_NAME = "labelling-{model_name}-rs{random_seed}"


def run_script(format_dict, seed, extra_arguments, partition=None, dependency=None, num_nodes=1):
    format_dict["random_seed"] = seed
    format_dict["extra_arguments"] = extra_arguments

    extra_sbatch_args = ["--parsable", f"--output={LOGS.format(**format_dict)}"]

    if dependency is not None:
        extra_sbatch_args.append(f"--dependency=afterany:{dependency}")

    job_id = launch_job(
        cmd=SLURM_CMD.format(**format_dict),
        num_nodes=num_nodes,
        tasks_per_node=format_dict["num_tasks"],
        gpus_per_node=format_dict["num_gpus"],
        job_name=JOB_NAME.format(**format_dict),
        container=CLUSTER_CONFIG["containers"][format_dict["server_type"]],
        mounts=MOUNTS.format(**format_dict),
        partition=partition,
        with_sandbox=True,
        extra_sbatch_args=extra_sbatch_args,
    )
    # going to return a previous job id
    return job_id


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--server_type", choices=('nemo', 'tensorrt_llm', 'vllm'), default='tensorrt_llm')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument(
        "--dependent_jobs",
        type=int,
        default=0,
        help="Specify this to launch that number of dependent jobs. Useful for large datasets, "
        "where you're not able to process everything before slurm timeout.",
    )
    parser.add_argument("--starting_seed", type=int, default=0)
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--extra_eval_args",
        default="",
        help="Any extra arguments to pass to nemo_skills/evaluation/evaluate_results.py",
    )
    args, unknown = parser.parse_known_args()

    args.model_path = Path(args.model_path).absolute()
    args.output_dir = Path(args.output_dir).absolute()

    extra_arguments = f'{" ".join(unknown)}'

    server_start_cmd, num_tasks, server_wait_string = get_server_command(
        args.server_type, args.num_gpus, args.num_nodes, args.model_path.name
    )

    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "output_dir": args.output_dir,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "num_tasks": num_tasks,
        "server_type": args.server_type,
        "extra_eval_args": args.extra_eval_args,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # needed for some of the models, so making an option to pass it in
        "server_wait_string": server_wait_string,
    }

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    for seed in range(args.starting_seed, args.starting_seed + args.num_runs):
        job_id = run_script(format_dict, seed, extra_arguments, args.partition, num_nodes=args.num_nodes)
        print(f"Submitted batch job {job_id}")
        for _ in range(args.dependent_jobs):
            job_id = run_script(
                format_dict,
                seed,
                extra_arguments,
                args.partition,
                dependency=job_id,
                num_nodes=args.num_nodes,
            )
            print(f"Submitted batch job {job_id}")
