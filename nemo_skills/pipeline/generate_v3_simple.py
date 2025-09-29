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

"""
Ultra-simple declarative generation CLI.

The CLI is now purely declarative - it just instantiates components
and the system handles all the complexity.
"""

from typing import List, Optional

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import get_cluster_config
from nemo_skills.pipeline.utils.declarative import (
    GenerateTask,
    HardwareConfig,
    HetGroup,
    Pipeline,
    Sandbox,
    Server,
)
from nemo_skills.utils import compute_chunk_ids, setup_logging, str_ids_to_list


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def generate_v3_simple(
    ctx: typer.Context,
    cluster: str = typer.Option(None, help="Cluster configuration"),
    input_file: str = typer.Option(None, help="Input data file"),
    input_dir: str = typer.Option(None, help="Input data directory"),
    output_dir: str = typer.Option(..., help="Output directory"),
    expname: str = typer.Option("generate", help="Experiment name"),
    model: str = typer.Option(None, help="Model path or name"),
    server_type: str = typer.Option("vllm", help="Server type"),
    server_gpus: int = typer.Option(None, help="Number of GPUs for server"),
    server_nodes: int = typer.Option(1, help="Number of nodes for server"),
    server_args: str = typer.Option("", help="Server arguments"),
    with_sandbox: bool = typer.Option(False, help="Include sandbox"),
    num_random_seeds: int = typer.Option(None, help="Number of random seeds"),
    random_seeds: str = typer.Option(None, help="Random seeds list"),
    starting_seed: int = typer.Option(0, help="Starting seed"),
    num_chunks: int = typer.Option(None, help="Number of chunks"),
    chunk_ids: str = typer.Option(None, help="Chunk IDs"),
    rerun_done: bool = typer.Option(False, help="Rerun completed jobs"),
    partition: str = typer.Option(None, help="Slurm partition"),
    exclusive: bool = typer.Option(False, help="Exclusive access"),
    dry_run: bool = typer.Option(False, help="Dry run mode"),
):
    """Generate outputs using ultra-simple declarative syntax."""

    setup_logging()

    # Parse seeds and chunks
    seeds = parse_seeds(random_seeds, num_random_seeds, starting_seed)
    chunks = parse_chunks(chunk_ids, num_chunks)

    # Get input source
    input_source = input_file or input_dir
    if not input_source:
        raise ValueError("Must specify either input_file or input_dir")

    # Get cluster config
    cluster_config = get_cluster_config(cluster)

    # PURE DECLARATIVE APPROACH - Just describe what you want!

    if server_gpus:
        # Self-hosted server case
        pipeline = Pipeline(
            name=expname,
            groups=[
                HetGroup(
                    [
                        server := Server(
                            model=model,
                            server_type=server_type,
                            gpus=server_gpus,
                            nodes=server_nodes,
                            args=server_args,
                        ),
                        sandbox := Sandbox() if with_sandbox else None,
                        GenerateTask(
                            input_source=input_source,
                            output_dir=output_dir,
                            server=server,
                            sandbox=sandbox,
                            seeds=seeds,
                            chunks=chunks,
                            rerun_done=rerun_done,
                            extra_args=ctx.args,  # Includes ++prompt_config etc.
                        ),
                    ],
                    hardware=HardwareConfig(
                        partition=partition,
                        exclusive=exclusive,
                        server_gpus=server_gpus,
                        server_nodes=server_nodes,
                    ),
                ).named("generation_with_server")
            ],
        )
    else:
        # External server case
        pipeline = Pipeline(
            name=expname,
            groups=[
                HetGroup(
                    [
                        sandbox := Sandbox() if with_sandbox else None,
                        GenerateTask(
                            input_source=input_source,
                            output_dir=output_dir,
                            sandbox=sandbox,
                            seeds=seeds,
                            chunks=chunks,
                            rerun_done=rerun_done,
                            extra_args=ctx.args,
                        ),
                    ],
                    hardware=HardwareConfig(
                        partition=partition,
                        exclusive=exclusive,
                    ),
                ).named("generation_external_server")
            ],
        )

    # Execute the declaratively defined pipeline
    return pipeline.run(cluster_config, dry_run=dry_run)


def parse_seeds(random_seeds: str, num_random_seeds: int, starting_seed: int) -> List[Optional[int]]:
    """Parse seed specifications."""
    if isinstance(random_seeds, str):
        return str_ids_to_list(random_seeds)
    elif num_random_seeds is not None:
        return list(range(starting_seed, starting_seed + num_random_seeds))
    return [None]


def parse_chunks(chunk_ids: str, num_chunks: int) -> List[Optional[int]]:
    """Parse chunk specifications."""
    if isinstance(chunk_ids, str):
        return str_ids_to_list(chunk_ids)
    elif num_chunks is not None:
        return compute_chunk_ids(num_chunks)
    return [None]


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
