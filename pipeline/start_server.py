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

import atexit
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, get_server_command, launch_job

from nemo_skills.utils import setup_logging

SLURM_CMD = """
nvidia-smi && \
export PYTHONPATH=/code && \
export HF_TOKEN={HF_TOKEN} && \
if [ $SLURM_PROCID -eq 0 ]; then \
    {{ {server_start_cmd} 2>&1 | tee /tmp/server_logs.txt & }} && sleep 1 && \
    echo "Waiting for the server to start" && \
    tail -n0 -f /tmp/server_logs.txt | sed '/{server_wait_string}/ q' && \
    tail -n10 /tmp/server_logs.txt &&  \
    SERVER_ADDRESS=$(tail -n 10 /tmp/server_logs.txt | \
    grep -oP 'http://\K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tail -n1) && \
    echo "Server is running on $SERVER_ADDRESS" && \
    {sandbox_echo} \
    sleep infinity; \
else \
    {server_start_cmd}; \
fi \
"""
MOUNTS = "{NEMO_SKILLS_CODE}:/code,{model_path}:/model"
JOB_NAME = "interactive-server-{server_type}-{model_name}"

# TODO: nemo does not exit on ctrl+c, need to fix that


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--server_type", choices=('nemo', 'tensorrt_llm', 'vllm'), default='tensorrt_llm')
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes required for hosting LLM server.",
    )
    parser.add_argument(
        "--partition",
        required=False,
        help="Can specify if need interactive jobs or a specific non-default partition",
    )
    parser.add_argument(
        "--no_sandbox", action="store_true", help="Disables sandbox if code execution is not required."
    )
    args = parser.parse_args()

    args.model_path = Path(args.model_path).absolute()

    server_start_cmd, num_tasks, server_wait_string = get_server_command(
        args.server_type, args.num_gpus, args.num_nodes, args.model_path.name
    )

    # TODO: VLLM
    sandbox_echo = 'echo "Sandbox is running on {$NEMO_SKILLS_SANDBOX_HOST:-$SERVER_ADDRESS}" &&'
    format_dict = {
        "model_path": args.model_path,
        "model_name": args.model_path.name,
        "num_gpus": args.num_gpus,
        "server_start_cmd": server_start_cmd,
        "server_type": args.server_type,
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),  # needed for some of the models, so making an option to pass it in
        "server_wait_string": server_wait_string,
        "sandbox_echo": sandbox_echo if not args.no_sandbox else "",
    }

    job_id = launch_job(
        cmd=SLURM_CMD.format(**format_dict),
        num_nodes=args.num_nodes,
        tasks_per_node=num_tasks,
        gpus_per_node=format_dict["num_gpus"],
        job_name=JOB_NAME.format(**format_dict),
        container=CLUSTER_CONFIG["containers"][args.server_type],
        mounts=MOUNTS.format(**format_dict),
        partition=args.partition,
        with_sandbox=(not args.no_sandbox),
        extra_sbatch_args=["--parsable"],
    )

    # the rest is only applicable for slurm execution - local execution will block on the launch_job call
    if CLUSTER_CONFIG["cluster"] != "slurm":
        sys.exit(0)

    log_file = f"slurm-{job_id}.out"

    # killing the serving job when exiting this script
    atexit.register(
        lambda job_id: subprocess.run(f"scancel {job_id}", shell=True, check=True),
        job_id,
    )
    # also cleaning up logs
    atexit.register(lambda log_file: os.remove(log_file), log_file)

    print("Please wait while the server is starting!")
    server_host = None
    server_started = False
    while True:  # waiting for the server to start
        time.sleep(1)
        # checking the logs to see if server has started
        if not os.path.isfile(log_file):
            continue
        with open(log_file) as fin:
            for line in fin:
                if "running on node" in line:
                    server_host = line.split()[-1].strip()
                if "Running on all addresses" in line:
                    server_started = True
        if server_started:
            print(f"Server has started at {server_host}")
            break
    print("Streaming server logs")
    while True:  # waiting for the kill signal and streaming logs
        subprocess.run(f"tail -f {log_file}", shell=True, check=True)
