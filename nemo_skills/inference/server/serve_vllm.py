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

import argparse
import os
import socket
import subprocess


def get_slurm_info():
    nodelist = os.environ.get('SLURM_STEP_NODELIST') or os.environ.get('SLURM_JOB_NODELIST')
    master_addr = (
        subprocess.check_output(f'scontrol show hostnames {nodelist} | head -n1', shell=True).decode().strip()
    )
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    return master_addr, local_rank, world_size


def main():
    parser = argparse.ArgumentParser(description="Serve vLLM model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args, unknown = parser.parse_known_args()

    master_addr, local_rank, world_size = get_slurm_info()
    is_master = local_rank == 0
    dist_port = args.port + 1  # Use separate port for distributed communication

    extra_arguments = f'{" ".join(unknown)}'

    if is_master:
        print(f"Deploying model {args.model} across {world_size} nodes")
        print("Starting OpenAI Server")

    cmd = (
        f'python -m vllm.entrypoints.openai.api_server '
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}"'
        f'    --trust-remote-code '
        f'    --host="0.0.0.0" '
        f'    --port={args.port} '
        f'    --tensor-parallel-size={args.num_gpus * world_size} '
        f'    --gpu-memory-utilization=0.9 '
        f'    --max-num-seqs=256 '
        f'    --enforce-eager '
        f'    --disable-log-requests '
        f'    --disable-log-stats '
        f'    --master-addr={master_addr} '
        f'    --master-port={dist_port} '
        f'    --worker-use-ray '
        f'    {extra_arguments} | grep -v "200 OK"'
    )

    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
