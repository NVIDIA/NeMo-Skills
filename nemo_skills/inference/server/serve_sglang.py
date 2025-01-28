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
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Serve STlang model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--node_rank", type=int, required=True)
    parser.add_argument("--dist_init_addr", type=str, required=True)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    args, unknown = parser.parse_known_args()

    extra_arguments = f'{" ".join(unknown)}'

    print(f"Deploying model {args.model}")
    print("Starting OpenAI Server")

    if args.verbose:
        logging_args = ""
    else:
        logging_args = ' --disable-log-requests --disable-log-stats '

    cmd = (
        f'python3 -m sglang.launch_server '
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}"'
        f'    --trust-remote-code '
        f'    --host="0.0.0.0" '
        f'    --port={args.port} '
        f'    --tensor-parallel-size={args.num_gpus*args.num_nodes} '
        f'    --nnodes={args.num_nodes} '
        f'    --node-rank={args.node_rank} '
        f'    --dist-init-addr="{args.dist_init_addr}:20000" '
    )

    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
