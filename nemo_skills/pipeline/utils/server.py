# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

from nemo_skills.pipeline.utils.mounts import check_if_mounted
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedServersSelfHosted(str, Enum):
    trtllm = "trtllm"
    trtllm_serve = "trtllm-serve"
    vllm = "vllm"
    nemo = "nemo"
    sglang = "sglang"
    megatron = "megatron"


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    trtllm_serve = "trtllm-serve"
    vllm = "vllm"
    nemo = "nemo"
    sglang = "sglang"
    megatron = "megatron"
    openai = "openai"
    azureopenai = "azureopenai"


def get_free_port(exclude: list[int] | None = None, strategy: int | str = 5000) -> int:
    """Will return a free port on the host."""
    exclude = exclude or []
    if isinstance(strategy, int):
        port = strategy
        while port in exclude:
            port += 1
        return port
    elif strategy == "random":
        import random

        port = random.randint(1024, 65535)
        while port in exclude:
            port = random.randint(1024, 65535)
        return port
    else:
        raise ValueError(f"Strategy {strategy} not supported.")


def should_get_random_port(server_gpus, exclusive, server_type):
    return server_gpus != 8 and not exclusive and server_type != "megatron"


def wait_for_server(server_address, generation_commands):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        # might be required if we are not hosting server ourselves
        # this will try to handshake in a loop and unblock when the server responds
        f"echo 'Waiting for the server to start at {server_address}' && "
        f"while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done && "
        # will run in a single task always (no need to check mpi env vars)
        f"{generation_commands}"
    )
    return cmd


def get_reward_server_command(
    server_type: str,
    num_gpus: int,
    num_nodes: int,
    model_path: str,
    cluster_config: dict,
    server_port: int,
    server_args: str = "",
    server_entrypoint: str | None = None,
):
    num_tasks = num_gpus

    # check if the model path is mounted if not vllm;
    # vllm can also pass model name as "model_path" so we need special processing
    if server_type != "vllm":
        check_if_mounted(cluster_config, model_path)

    # the model path will be mounted, so generally it will start with /
    elif server_type == "vllm" and model_path.startswith("/"):
        check_if_mounted(cluster_config, model_path)

    if server_type == 'nemo':
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_nemo_aligner_reward_model"
        nemo_aligner_reward_model_port = get_free_port(strategy="random", exclude=[server_port])
        server_start_cmd = (
            # Note: The order of the two commands is important as the reward model server
            # needs to be the first command so it can get the HF_TOKEN from the environment
            f"python -m {server_entrypoint} "
            f"    ++rm_model_file={model_path} "
            f"    trainer.devices={num_gpus} "
            f"    trainer.num_nodes={num_nodes} "
            f"    +model.tensor_model_parallel_size={num_gpus} "
            f"    +model.pipeline_model_parallel_size={num_nodes} "
            # This port could be configurable, but is hard coded to reduce
            # the divergence of the server command parameters from pipeline/generate.py
            f"    inference.port={nemo_aligner_reward_model_port} "
            f"    {server_args} & "
            f"python -m nemo_skills.inference.server.serve_nemo_reward_model "
            # These ports could be configurable, but is hard coded to reduce
            # the divergence of the server command parameters from pipeline/generate.py
            f"    inference_port={server_port}  "
            f"    triton_server_address=localhost:{nemo_aligner_reward_model_port} "
        )

        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] != "slurm":
            num_tasks = 1

    elif server_type == "vllm":
        if num_nodes > 1:
            raise ValueError("VLLM server does not support multi-node execution")

        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_vllm"
        server_start_cmd = (
            f"python3 -m {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        num_tasks = 1
    else:
        raise ValueError(f"Server type '{server_type}' not supported for reward model.")

    server_cmd = (
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_ray_server_cmd(start_cmd):
    ports = (
        "--node-manager-port=12345 "
        "--object-manager-port=12346 "
        "--dashboard-port=8265 "
        "--dashboard-agent-grpc-port=12347 "
        "--runtime-env-agent-port=12349 "
        "--metrics-export-port=12350 "
        "--min-worker-port=14349 "
        "--max-worker-port=18349 "
    )

    ray_start_cmd = (
        "if [ \"${SLURM_PROCID:-0}\" = 0 ]; then "
        "    echo 'Starting head node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    ray start "
        "        --head "
        "        --port=6379 "
        f"       {ports} && "
        f"   {start_cmd} ; "
        "else "
        "    echo 'Starting worker node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    echo \"Connecting to head node at $SLURM_MASTER_NODE\" && "
        "    ray start "
        "        --block "
        "        --address=$SLURM_MASTER_NODE:6379 "
        f"       {ports} ;"
        "fi"
    )
    return ray_start_cmd


def get_server_command(
    server_type: str,
    num_gpus: int,
    num_nodes: int,
    model_path: str,
    cluster_config: dict,
    server_port: int,
    server_args: str = "",
    server_entrypoint: str | None = None,
):
    num_tasks = num_gpus

    # check if the model path is mounted if not vllm;
    # vllm can also pass model name as "model_path" so we need special processing
    if server_type not in ["vllm", "sglang", "trtllm-serve"]:
        check_if_mounted(cluster_config, model_path)

    # the model path will be mounted, so generally it will start with /
    elif model_path.startswith("/"):
        check_if_mounted(cluster_config, model_path)

    if server_type == 'nemo':
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_nemo"
        server_start_cmd = (
            f"python {server_entrypoint} "
            f"    gpt_model_file={model_path} "
            f"    trainer.devices={num_gpus} "
            f"    trainer.num_nodes={num_nodes} "
            f"    tensor_model_parallel_size={num_gpus} "
            f"    pipeline_model_parallel_size={num_nodes} "
            f"    ++port={server_port} "
            f"    {server_args} "
        )

        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] != "slurm":
            num_tasks = 1
    elif server_type == 'megatron':
        if cluster_config["executor"] != "slurm":
            num_tasks = 1
            prefix = f"torchrun --nproc_per_node {num_gpus}"
        else:
            prefix = "python "
        server_entrypoint = server_entrypoint or "tools/run_text_generation_server.py"
        # similar to conversion, we don't hold scripts for megatron on our side
        # and expect it to be in /opt/Megatron-LM in the container
        server_start_cmd = (
            f"export PYTHONPATH=$PYTHONPATH:/opt/Megatron-LM && "
            f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
            f"cd /opt/Megatron-LM && "
            f"{prefix} {server_entrypoint} "
            f"    --load {model_path} "
            f"    --tensor-model-parallel-size {num_gpus} "
            f"    --pipeline-model-parallel-size {num_nodes} "
            f"    --use-checkpoint-args "
            f"    --max-tokens-to-oom 12000000 "
            f"    --micro-batch-size 1 "  # that's a training argument, ignored here, but required to specify..
            f"    {server_args} "
        )
    elif server_type == 'vllm':
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_vllm"
        start_vllm_cmd = (
            f"python3 {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        if num_nodes > 1:
            server_start_cmd = get_ray_server_cmd(start_vllm_cmd)
        else:
            server_start_cmd = start_vllm_cmd
        num_tasks = 1
    elif server_type == 'sglang':
        if num_nodes > 1:
            multinode_args = f" --dist_init_addr $SLURM_MASTER_NODE --node_rank $SLURM_PROCID "
        else:
            multinode_args = ""
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_sglang"
        server_start_cmd = (
            f"python3 {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {multinode_args} "
            f"    {server_args} "
        )
        num_tasks = 1
    elif server_type == 'trtllm':
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_trt"
        server_start_cmd = (
            f"FORCE_NCCL_ALL_REDUCE_STRATEGY=1 python -m {server_entrypoint} "
            f"    --model_path {model_path} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        num_tasks = num_gpus
    elif server_type == 'trtllm-serve':
        server_entrypoint = server_entrypoint or "trtllm-serve"
        if num_nodes > 1 and server_entrypoint == "trtllm-serve":
            server_entrypoint = f"trtllm-llmapi-launch {server_entrypoint}"
        server_start_cmd = (
            f"{server_entrypoint} "
            f"    {model_path} "
            f"    --port {server_port} "
            f"    --tp_size {num_gpus * num_nodes} "
            f"    {server_args} "
        )
        if num_nodes == 1:
            num_tasks = 1
        else:
            num_tasks = num_gpus
    else:
        raise ValueError(f"Server type '{server_type}' not supported for model inference.")

    server_cmd = (
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks
