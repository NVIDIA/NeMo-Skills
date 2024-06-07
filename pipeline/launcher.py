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

# Wraps up slurm sbatch scripts to be able to seamlessly switch between
# different clusters and local execution

import argparse
import atexit
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from time import sleep

import yaml

LOG = logging.getLogger(__file__)
WRAPPER_HELP = """
This is a wrapper script that will help to launch a pipeline job. You can find the job configuration
arguments under the "wrapper arguments" section. You can also customize any of the arguments of
the underlying script, which are listed under the "script arguments" section.
""".strip()


def fill_env_vars(format_dict, env_vars):
    for env_var in env_vars:
        env_var_value = os.getenv(env_var)
        if not env_var_value:
            raise ValueError(f"Must provide {env_var} environment variable")

        format_dict[env_var] = env_var_value


def get_server_command(server_type: str, num_gpus: int, num_nodes: int, model_name: str):
    num_tasks = num_gpus
    if server_type == 'nemo':
        server_start_cmd = (
            f"python /code/nemo_skills/inference/server/serve_nemo.py gpt_model_file=/model "
            f"trainer.devices={num_gpus} "
            f"trainer.num_nodes={num_nodes} "
            f"tensor_model_parallel_size={num_gpus} "
            f"pipeline_model_parallel_size={num_nodes} "
        )
        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if CLUSTER_CONFIG["cluster"] == "local":
            num_tasks = 1

    elif server_type == 'vllm':
        server_start_cmd = (
            f"NUM_GPUS={num_gpus} bash /code/nemo_skills/inference/server/serve_vllm.sh "
            f"/model/ {model_name} 0 openai 5000"
        )
        num_tasks = 1
    else:
        # adding sleep to ensure the logs file exists
        server_start_cmd = f"python /code/nemo_skills/inference/server/serve_trt.py --model_path /model"
        num_tasks = num_gpus
    if server_type == "vllm":
        server_wait_string = "Uvicorn running"
    else:
        server_wait_string = "Running on all addresses"
    return server_start_cmd, num_tasks, server_wait_string


SLURM_HEADER = """
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH -N {num_nodes}
#SBATCH -t {timeout}
#SBATCH -J "{job_name_prefix}:{job_name}"
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
"""

NEMO_SKILLS_CODE = str(Path(__file__).absolute().parents[1])
config_file = os.getenv("NEMO_SKILLS_CONFIG")
if not config_file:
    raise ValueError("Must provide NEMO_SKILLS_CONFIG environment variable")

with open(config_file, "rt", encoding="utf-8") as fin:
    CLUSTER_CONFIG = yaml.safe_load(fin)

SLURM_HEADER = CLUSTER_CONFIG.get("slurm_header", SLURM_HEADER)


def launch_local_job(
    cmd,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    container,
    mounts,
    partition=None,
    with_sandbox=False,
    extra_sbatch_args=None,
):
    if num_nodes > 1:
        raise ValueError("Local execution does not support multiple nodes")

    if partition is not None:
        raise ValueError("Local execution does not support partition specification")

    if extra_sbatch_args is not None:
        LOG.warning("Local execution does not support extra sbatch args. Got %s", str(extra_sbatch_args))

    mounts = " -v ".join(mounts.split(","))
    if mounts:
        mounts = f"-v {mounts}"

    cmd = cmd.strip()
    cmd = (
        f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(gpus_per_node)))} && "
        f"export SLURM_LOCALID={'$OMPI_COMM_WORLD_LOCAL_RANK' if tasks_per_node > 1 else 0} && "
        f"export SLURM_PROCID={'$OMPI_COMM_WORLD_LOCAL_RANK' if tasks_per_node > 1 else 0} && "
        f"{cmd}"
    )

    docker_cmd = CLUSTER_CONFIG["docker_cmd"]
    if with_sandbox:
        sandbox_name = f"local-sandbox-{uuid.uuid4()}"
        sandbox_cmd = (
            f'cd {Path(__file__).parents[1]} && '
            f'./nemo_skills/code_execution/local_sandbox/start_local_sandbox.sh {sandbox_name}'
        )
        sandbox_process = subprocess.Popen(sandbox_cmd, shell=True)
        atexit.register(sandbox_process.kill)

        # waiting for the sandbox to start and getting the host address
        get_host_cmd = (
            docker_cmd
            + " inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' `"
            + docker_cmd
            + f" ps -a | grep {sandbox_name}  | awk '{{print $1}}'`"
        )
        LOG.info("Waiting for the sandbox to start...")
        while True:
            try:
                output = subprocess.run(get_host_cmd, shell=True, check=True, capture_output=True)
                host = output.stdout.decode().strip()
                LOG.info(f"Sandbox started at {host}")
            except subprocess.CalledProcessError:
                sleep(5)
                continue
            else:
                break

        cmd = f"export NEMO_SKILLS_SANDBOX_HOST={host} && {cmd}"

    with tempfile.NamedTemporaryFile(mode="wt", delete=False) as fp:
        fp.write(cmd)
    LOG.info("Running command %s", cmd)
    mounts += f" -v {fp.name}:/start.sh"

    if tasks_per_node > 1:
        start_cmd = f'mpirun --allow-run-as-root -np {tasks_per_node} bash /start.sh'
    else:
        start_cmd = "bash /start.sh"

    cmd = f"{docker_cmd} run --rm --gpus all --ipc=host {mounts} {container} bash -c '{start_cmd}'"
    subprocess.run(cmd, shell=True, check=True)


def launch_slurm_job(
    cmd,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    container,
    mounts,
    partition=None,
    with_sandbox=False,
    extra_sbatch_args=None,
):
    partition = partition or CLUSTER_CONFIG["partition"]
    # compiling extra arguments from command line, config and env var
    extra_sbatch_args = extra_sbatch_args or []
    extra_sbatch_args += CLUSTER_CONFIG.get("extra_sbatch_args", [])
    if os.getenv("EXTRA_SBATCH_ARGS"):
        extra_sbatch_args += os.getenv("EXTRA_SBATCH_ARGS").split(" ")

    if 'timeouts' not in CLUSTER_CONFIG:
        timeout = "10000:00:00:00"
    else:
        timeout = CLUSTER_CONFIG["timeouts"][partition]

    header = SLURM_HEADER.format(
        account=CLUSTER_CONFIG["account"],
        partition=partition,
        num_nodes=num_nodes,
        timeout=timeout,
        job_name_prefix=CLUSTER_CONFIG["job_name_prefix"],
        job_name=job_name,
        tasks_per_node=tasks_per_node,
        gpus_per_node=gpus_per_node,
    )
    header += "\n".join([f"#SBATCH {arg}" for arg in extra_sbatch_args])

    cmd = cmd.replace("$", "\\$")
    cmd = f"""#!/bin/bash
{header}
set -x

echo running on node $(hostname)

read -r -d '' cmd <<EOF
{cmd}
EOF
"""
    if with_sandbox:
        # we should estimate optimal memory and cpu requirements for sandbox
        # right now splitting half-and-half, since both evaluation and training
        # do not use CPUs or CPU-RAM that much

        extra_main_args = ""
        extra_sandbox_args = " ".join(CLUSTER_CONFIG.get("extra_sandbox_args", []))
        try:
            max_cpus = CLUSTER_CONFIG["max_cpus"][partition]
            max_memory = CLUSTER_CONFIG["max_memory"][partition]
            extra_main_args += f" --cpus-per-task={max_cpus // (2 * tasks_per_node)} --mem={max_memory // 2}M "
            extra_sandbox_args += f" --cpus-per-task={max_cpus // 2} --mem={max_memory // 2}M "
        except KeyError:
            pass
        cmd += f"""
srun {extra_main_args} --mpi=pmix --container-image={container} --container-mounts={mounts} bash -c "$cmd" &
MAIN_PID=$!
srun {extra_sandbox_args} --mpi=pmix --ntasks={num_nodes} \
     --container-image={CLUSTER_CONFIG["containers"]["sandbox"]} \
     bash -c "/entrypoint.sh && /start.sh" &
wait $MAIN_PID
"""

    # note how we are waiting only for the main command and sandbox will be killed when it finishes
    else:
        cmd = cmd + f'\nsrun --mpi=pmix --container-image={container} --container-mounts={mounts} bash -c "$cmd"'
    with tempfile.NamedTemporaryFile(mode="wt", delete=False) as fp:
        fp.write(cmd)
    try:
        result = subprocess.run(f"sbatch {fp.name}", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        LOG.error(exc.output)
        LOG.error(exc.stderr)
        raise
    else:
        # using simple print here since other process might rely on reading this
        print(result.stdout.decode())
    os.unlink(fp.name)

    return result.stdout.decode().strip()


launch_map = {
    'slurm': launch_slurm_job,
    'local': launch_local_job,
}
launch_job = launch_map[CLUSTER_CONFIG['cluster']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="Full command for cluster execution")
    parser.add_argument("--partition", required=False)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--tasks_per_node", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--gpus_per_node", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--with_sandbox", action="store_true")
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--mounts", required=True)
    args, unknown = parser.parse_known_args()

    launch_job(
        args.cmd,
        args.num_nodes,
        args.tasks_per_node,
        args.gpus_per_node,
        args.job_name,
        args.container,
        args.mounts,
        args.partition,
        args.with_sandbox,
        unknown,
    )
