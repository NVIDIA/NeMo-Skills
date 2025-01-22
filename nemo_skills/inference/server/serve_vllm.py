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
import subprocess

import ray


def main():
    parser = argparse.ArgumentParser(description="Serve vLLM model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    print(f"Deploying model {args.model}")
    print("Starting OpenAI Server")

    # TODO: don't break local
    # Get node information from SLURM env vars
    node_rank = int(os.environ["SLURM_PROCID"])
    head_node = os.environ["SLURM_NODELIST"].split(",")[0]
    print(f"Node rank: {node_rank}, head node: {head_node}")
    print(f"All nodes: {os.environ['SLURM_NODELIST']}")

    # Initialize Ray based on node rank
    if node_rank == 0:
        print("I'm the head node", flush=True)
        ray.init(_node_ip_address=head_node)
        print("Head node is done!", flush=True)
    else:
        import time

        time.sleep(10)
        print("I'm a worker node", flush=True)
        ray.init(address=f"ray://{head_node}:6379")
        print("Worker is done!", flush=True)

    # cmd = (
    #     f'python -m vllm.entrypoints.openai.api_server '
    #     f'    --model="{args.model}" '
    #     f'    --served-model-name="{args.model}"'
    #     f'    --trust-remote-code '
    #     f'    --host="0.0.0.0" '
    #     f'    --port={args.port} '
    #     f'    --tensor-parallel-size={args.num_gpus} '
    #     f'    --gpu-memory-utilization=0.9 '
    #     f'    --max-num-seqs=256 '
    #     f'    --enforce-eager '
    #     f'    --disable-log-requests '
    #     f'    --disable-log-stats '
    #     f'    {extra_arguments} | grep -v "200 OK"'
    # )

    # subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
