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

"""Script subclasses for declarative pipeline interface.

These Script subclasses encapsulate component-specific logic and expose typed
fields instead of using metadata dicts. Each Script knows how to build its own
command and environment.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import nemo_run as run

from nemo_skills.pipeline.utils.exp import get_sandbox_command
from nemo_skills.pipeline.utils.generation import get_generation_cmd
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command


@dataclass
class LLMServerScript(run.Script):
    """Script for running LLM servers (vLLM, SGLang, TRT-LLM, Megatron, etc).

    Attributes:
        model: Model path or name
        port: Port to serve on (default: random free port)
        gpus: Number of GPUs per node
        nodes: Number of nodes
        server_args: Additional arguments to pass to server
        server_type: Type of server (vllm, sglang, trtllm, megatron, generic)
        server_entrypoint: Custom entrypoint script path
        cluster_config: Cluster configuration dict
    """

    model: str = ""
    port: int = field(default_factory=lambda: get_free_port(strategy="random"))
    gpus: int = 8
    nodes: int = 1
    server_args: str = ""
    server_type: str = "vllm"
    server_entrypoint: Optional[str] = None
    cluster_config: dict = field(default_factory=dict)

    @property
    def num_tasks(self) -> int:
        """Number of tasks per node (computed from server type logic)."""
        # This mirrors the logic in get_server_command
        if self.server_type in ["vllm", "sglang"]:
            return 1
        elif self.server_type == "trtllm":
            return 1 if self.nodes == 1 else self.gpus
        elif self.server_type == "megatron":
            if self.cluster_config.get("executor") != "slurm":
                return 1
            return self.gpus
        elif self.server_type == "generic":
            return 1
        return self.gpus

    @property
    def log_prefix(self) -> str:
        """Prefix for log files."""
        return "server"

    def __post_init__(self):
        if not self.model:
            raise ValueError("model cannot be undefined.")

        # Get the server command using existing utility
        cmd, num_tasks = get_server_command(
            server_type=self.server_type,
            num_gpus=self.gpus,
            num_nodes=self.nodes,
            model_path=self.model,
            cluster_config=self.cluster_config,
            server_port=self.port,
            server_args=self.server_args,
            server_entrypoint=self.server_entrypoint,
        )

        self.inline = cmd

        # Set environment variables (these will be merged with cluster env vars)
        if not self.env:
            self.env = {}

        super().__post_init__()


@dataclass
class SandboxScript(run.Script):
    """Script for running code execution sandbox.

    Attributes:
        port: Port to serve on (required)
        cluster_config: Cluster configuration dict
    """

    port: int = 0
    cluster_config: dict = field(default_factory=dict)

    @property
    def log_prefix(self) -> str:
        """Prefix for log files."""
        return "sandbox"

    def __post_init__(self):
        if not self.port:
            raise ValueError("port must be specified for SandboxScript")

        # Get sandbox command
        self.inline = get_sandbox_command(self.cluster_config)

        # Build PYTHONPATH from cluster config
        pythonpath = None
        for env_var in self.cluster_config.get("env_vars", []):
            if "PYTHONPATH" in env_var:
                pythonpath = env_var[11:] if env_var.startswith("PYTHONPATH=") else env_var
                break

        # Set environment variables
        if not self.env:
            self.env = {}

        self.env["LISTEN_PORT"] = str(self.port)
        self.env["NGINX_PORT"] = str(self.port)
        if pythonpath:
            self.env["PYTHONPATH"] = pythonpath + ":/app"

        super().__post_init__()


@dataclass
class GenerationClientScript(run.Script):
    """Script for running generation client.

    Attributes:
        output_dir: Output directory for results
        input_file: Input file path (mutually exclusive with input_dir)
        input_dir: Input directory path (mutually exclusive with input_file)
        model_names: List of model names
        server_addresses: List of server addresses or callable returning addresses
        extra_arguments: Extra arguments to pass to generation script
        random_seed: Random seed for generation
        chunk_id: Chunk ID for parallel processing
        num_chunks: Total number of chunks
        preprocess_cmd: Command to run before generation
        postprocess_cmd: Command to run after generation
        wandb_parameters: Weights & Biases logging parameters
        script: Generation script module or path
        num_models: Number of models (computed from model_names if not provided)
    """

    output_dir: str = ""
    input_file: Optional[str] = None
    input_dir: Optional[str] = None
    model_names: list[str] = field(default_factory=list)
    server_addresses: Union[list[str], Callable[[], list[str]]] = field(default_factory=list)
    extra_arguments: str = ""
    random_seed: Optional[int] = None
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
    preprocess_cmd: Optional[str] = None
    postprocess_cmd: Optional[str] = None
    wandb_parameters: Optional[dict] = None
    script: str = "nemo_skills.inference.generate"
    num_models: Optional[int] = None

    @property
    def log_prefix(self) -> str:
        """Prefix for log files."""
        return "main"

    def __post_init__(self):
        if not self.output_dir:
            raise ValueError("output_dir must be specified for GenerationClientScript")

        # Resolve callable addresses if needed (for cross-component references)
        if callable(self.server_addresses):
            addresses = self.server_addresses()
        else:
            addresses = self.server_addresses

        # Compute num_models if not provided
        num_models = self.num_models or len(self.model_names)

        # Build generation command
        cmd = get_generation_cmd(
            output_dir=self.output_dir,
            input_file=self.input_file,
            input_dir=self.input_dir,
            extra_arguments=self.extra_arguments,
            random_seed=self.random_seed,
            chunk_id=self.chunk_id,
            num_chunks=self.num_chunks,
            preprocess_cmd=self.preprocess_cmd,
            postprocess_cmd=self.postprocess_cmd,
            wandb_parameters=self.wandb_parameters,
            script=self.script,
            server_addresses=addresses if addresses else None,
            model_names=self.model_names if self.model_names else None,
            num_models=num_models if num_models > 0 else None,
        )

        self.inline = cmd

        if not self.env:
            self.env = {}

        super().__post_init__()
