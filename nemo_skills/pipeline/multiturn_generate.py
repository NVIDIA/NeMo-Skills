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

"""
Multi-turn conversation generation between two models.

This script orchestrates a conversation between two language models by:
1. Starting two model servers (model_a and model_b) in a heterogeneous SLURM job
2. Running a generation client that alternates requests between the two servers
3. Outputting conversations in chat format to a single JSONL file

Example CLI usage:
    python -m nemo_skills.pipeline.multiturn_generate \\
        --cluster slurm \\
        --model-a nvidia/Llama-3-8B \\
        --model-b nvidia/Llama-3-70B \\
        --input-file /data/prompts.jsonl \\
        --output-dir /results/conversations \\
        --num-turns 4 \\
        --server-gpus-a 8 \\
        --server-gpus-b 8

Example Python usage:
    from nemo_skills.pipeline.multiturn_generate import multiturn_generate
    from nemo_skills.pipeline.utils import get_cluster_config

    cluster_config = get_cluster_config("slurm")

    multiturn_generate(
        cluster="slurm",
        model_a="nvidia/Llama-3-8B",
        model_b="nvidia/Llama-3-70B",
        input_file="/data/prompts.jsonl",
        output_dir="/results/conversations",
        num_turns=4,
        server_gpus_a=8,
        server_gpus_b=8,
    )
"""

import logging
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.commands import sandbox_command, vllm_server_command
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def _build_multiturn_generation_command(
    input_file: str,
    output_file: str,
    server_a_hostname: str,
    server_a_port: str,
    server_b_hostname: str,
    server_b_port: str,
    num_turns: int,
    model_a_name: str,
    model_b_name: str,
    extra_arguments: str = "",
) -> str:
    """Build the multi-turn generation command string."""
    return f"""
python -m nemo_skills.inference.multiturn_conversation \\
    ++input_file={input_file} \\
    ++output_file={output_file} \\
    ++server_a_address={server_a_hostname}:{server_a_port} \\
    ++server_b_address={server_b_hostname}:{server_b_port} \\
    ++num_turns={num_turns} \\
    ++model_a_name={model_a_name} \\
    ++model_b_name={model_b_name} \\
    {extra_arguments}
""".strip()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def multiturn_generate(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    model_a: str = typer.Option(..., help="Path to model A or model name (first speaker)"),
    model_b: str = typer.Option(..., help="Path to model B or model name (second speaker)"),
    input_file: str = typer.Option(..., help="Path to input JSONL file with initial prompts"),
    output_dir: str = typer.Option(..., help="Where to put conversation results"),
    num_turns: int = typer.Option(4, help="Number of conversation turns (total exchanges)"),
    expname: str = typer.Option("multiturn_conversation", help="Nemo run experiment name"),
    server_type_a: pipeline_utils.SupportedServers = typer.Option("vllm", help="Type of server to use for model A"),
    server_type_b: pipeline_utils.SupportedServers = typer.Option("vllm", help="Type of server to use for model B"),
    server_gpus_a: int = typer.Option(8, help="Number of GPUs for model A server"),
    server_gpus_b: int = typer.Option(8, help="Number of GPUs for model B server"),
    server_nodes_a: int = typer.Option(1, help="Number of nodes for model A server"),
    server_nodes_b: int = typer.Option(1, help="Number of nodes for model B server"),
    server_args_a: str = typer.Option("", help="Extra arguments to pass to model A server"),
    server_args_b: str = typer.Option("", help="Extra arguments to pass to model B server"),
    server_entrypoint_a: str = typer.Option(
        None, help="Path to the entrypoint of model A server (uses default if not specified)"
    ),
    server_entrypoint_b: str = typer.Option(
        None, help="Path to the entrypoint of model B server (uses default if not specified)"
    ),
    server_container_a: str = typer.Option(
        None, help="Override container image for model A server (uses cluster config default if not specified)"
    ),
    server_container_b: str = typer.Option(
        None, help="Override container image for model B server (uses cluster config default if not specified)"
    ),
    with_sandbox_a: bool = typer.Option(False, help="Start sandbox container for model A"),
    with_sandbox_b: bool = typer.Option(False, help="Start sandbox container for model B"),
    partition: str = typer.Option(None, help="Slurm partition to use"),
    qos: str = typer.Option(None, help="Slurm QoS to use"),
    time_min: str = typer.Option(None, help="Slurm time-min parameter"),
    exclusive: bool = typer.Option(False, help="Request exclusive node allocation"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    run_after: List[str] = typer.Option(
        None, help="List of experiment names that need to complete before this starts"
    ),
    reuse_code: bool = typer.Option(True, help="Reuse code from previous experiment"),
    reuse_code_exp: str = typer.Option(None, help="Specific experiment to reuse code from"),
    config_dir: str = typer.Option(None, help="Custom location for cluster configs"),
    log_dir: str = typer.Option(None, help="Custom location for slurm logs"),
    installation_command: str = typer.Option(
        None, help="Installation command to run before main job (only affects client, not servers)"
    ),
    skip_hf_home_check: bool = typer.Option(
        None, help="Skip checking that HF_HOME env var is defined in cluster config"
    ),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    dry_run: bool = typer.Option(False, help="Validate arguments without running the job"),
):
    """Generate multi-turn conversations between two models.

    This command starts two model servers and orchestrates a conversation where they
    take turns responding to each other. The output is a single JSONL file where each
    line contains a complete conversation in chat format.

    Run with --help to see additional arguments. Extra arguments (prefixed with ++)
    will be passed to the underlying inference script.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting multi-turn conversation generation")
    LOG.info("Model A: %s", model_a)
    LOG.info("Model B: %s", model_b)
    LOG.info("Number of turns: %d", num_turns)
    LOG.info("Extra arguments: %s", extra_arguments)

    # Handle server type enums
    try:
        server_type_a = server_type_a.value
    except AttributeError:
        pass
    try:
        server_type_b = server_type_b.value
    except AttributeError:
        pass

    # Prepare cluster config
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    if not log_dir:
        log_dir = f"{output_dir}/multiturn-logs"

    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    # Get random ports for servers
    get_random_port = pipeline_utils.should_get_random_port(max(server_gpus_a, server_gpus_b), exclusive)

    # Build server commands
    server_a_cmd, server_a_meta = vllm_server_command(
        cluster_config=cluster_config,
        model=model_a,
        port=None if get_random_port else 5000,
        server_type=server_type_a,
        gpus=server_gpus_a,
        nodes=server_nodes_a,
        args=server_args_a,
        entrypoint=server_entrypoint_a,
    )

    server_b_cmd, server_b_meta = vllm_server_command(
        cluster_config=cluster_config,
        model=model_b,
        port=None if get_random_port else 5001,
        server_type=server_type_b,
        gpus=server_gpus_b,
        nodes=server_nodes_b,
        args=server_args_b,
        entrypoint=server_entrypoint_b,
    )

    # Resolve containers
    container_a = server_container_a or cluster_config["containers"].get(
        server_type_a, cluster_config["containers"]["nemo-skills"]
    )
    container_b = server_container_b or cluster_config["containers"].get(
        server_type_b, cluster_config["containers"]["nemo-skills"]
    )

    # Create Command objects for model A group
    server_a = Command(
        command=server_a_cmd,
        container=container_a,
        gpus=server_gpus_a,
        nodes=server_nodes_a,
        name="server_a",
        metadata=server_a_meta,
    )

    components_a = [server_a]

    # Add sandbox for model A if requested
    if with_sandbox_a:
        sandbox_a_cmd, sandbox_a_meta = sandbox_command(
            cluster_config=cluster_config,
            port=None if get_random_port else 6000,
        )
        sandbox_a_meta["log_prefix"] = "sandbox_a"
        sandbox_a = Command(
            command=sandbox_a_cmd,
            container=cluster_config["containers"]["sandbox"],
            name="sandbox_a",
            metadata=sandbox_a_meta,
        )
        components_a.append(sandbox_a)

    # Multi-turn generation client (runs in group A, talks to both servers)
    # Use lambda for cross-component hostname references
    def build_client_command():
        return _build_multiturn_generation_command(
            input_file=input_file,
            output_file=f"{output_dir}/conversations.jsonl",
            server_a_hostname=server_a.hostname_ref(),
            server_a_port=server_a.meta_ref("port"),
            server_b_hostname=server_b.hostname_ref(),
            server_b_port=server_b.meta_ref("port"),
            num_turns=num_turns,
            model_a_name=model_a,
            model_b_name=model_b,
            extra_arguments=extra_arguments,
        )

    client = Command(
        command=build_client_command,
        container=cluster_config["containers"]["nemo-skills"],
        name="multiturn_client",
        installation_command=installation_command,
        metadata={"log_prefix": "main"},
    )
    components_a.append(client)

    # Create CommandGroup for model A + client
    group_a = CommandGroup(
        commands=components_a,
        hardware=HardwareConfig(
            partition=partition,
            qos=qos,
            time_min=time_min,
            exclusive=exclusive,
            num_gpus=server_gpus_a,
            num_nodes=server_nodes_a,
        ),
        name="model_a_group",
        log_dir=log_dir,
    )

    # Create Command objects for model B group
    server_b = Command(
        command=server_b_cmd,
        container=container_b,
        gpus=server_gpus_b,
        nodes=server_nodes_b,
        name="server_b",
        metadata=server_b_meta,
    )

    components_b = [server_b]

    # Add sandbox for model B if requested
    if with_sandbox_b:
        sandbox_b_cmd, sandbox_b_meta = sandbox_command(
            cluster_config=cluster_config,
            port=None if get_random_port else 6001,
        )
        sandbox_b_meta["log_prefix"] = "sandbox_b"
        sandbox_b = Command(
            command=sandbox_b_cmd,
            container=cluster_config["containers"]["sandbox"],
            name="sandbox_b",
            metadata=sandbox_b_meta,
        )
        components_b.append(sandbox_b)

    # Create CommandGroup for model B
    group_b = CommandGroup(
        commands=components_b,
        hardware=HardwareConfig(
            partition=partition,
            qos=qos,
            time_min=time_min,
            exclusive=exclusive,
            num_gpus=server_gpus_b,
            num_nodes=server_nodes_b,
        ),
        name="model_b_group",
        log_dir=log_dir,
    )

    # Create pipeline with heterogeneous job (both groups in one SLURM job)
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=[
            {
                "name": "multiturn_conversation",
                "groups": [group_a, group_b],  # Heterogeneous job
                "dependencies": run_after if run_after else None,
            }
        ],
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=skip_hf_home_check,
    )

    # Run pipeline
    result = pipeline.run(dry_run=dry_run)

    if not dry_run:
        LOG.info("Multi-turn conversation job submitted successfully")
        LOG.info(f"Output will be written to: {output_dir}/conversations.jsonl")

    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
