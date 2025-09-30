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
Truly declarative pipeline system where you instantiate components and define relationships.

Example usage:
    server = Server(model="Qwen/Qwen3-8B", server_type="vllm", gpus=2)
    sandbox = Sandbox(port=6000)
    generation = GenerateTask(
        input_file="data.jsonl",
        output_dir="results",
        server=server,
        sandbox=sandbox
    )

    pipeline = Pipeline(groups=[
        HetGroup([server, sandbox, generation])
    ])
    pipeline.run(cluster_config)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from nemo_skills.pipeline.utils import get_exp, run_exp
from nemo_skills.pipeline.utils.generation import get_chunked_rs_filename
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.pipeline.utils.task_factories import TaskFactory
from nemo_skills.pipeline.utils.task_system import PipelineBuilder, TaskGroup
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class RuntimeRef:
    """Reference to a component that will be resolved at runtime using SLURM environment variables."""

    component_name: str
    het_group_index: int
    attribute: str = "host"  # What to resolve (host, port, etc.)

    def resolve_expression(self, cluster_config: Optional[Dict] = None) -> str:
        """Get the runtime expression to resolve this reference."""
        if self.attribute == "host":
            # Check execution environment
            if cluster_config and cluster_config.get("executor") in ["local", "none"]:
                # Local execution - everything runs on localhost
                return "127.0.0.1"
            else:
                # SLURM execution - use environment variable to get first host in the het group
                return f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{self.het_group_index} | head -n1)"
        elif self.attribute == "port":
            # Use environment variable set by the component
            return f"${{{self.component_name.upper()}_PORT_HET_GROUP_{self.het_group_index}}}"
        else:
            return f"${{{self.component_name.upper()}_{self.attribute.upper()}_HET_GROUP_{self.het_group_index}}}"

    def __str__(self):
        return f"RuntimeRef({self.component_name}.{self.attribute} from group {self.het_group_index})"


# Base class for all declarative components
class Component:
    """Base class for all pipeline components - all are cross-group compatible by default."""

    def __init__(self):
        self.het_group_index: Optional[int] = None  # Set automatically by Pipeline

    def get_runtime_ref(self, attribute: str = "host") -> RuntimeRef:
        """Get a runtime reference to this component for cross-group access."""
        return RuntimeRef(component_name=self.get_name(), het_group_index=self.het_group_index, attribute=attribute)

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert this component to a TaskDefinition."""
        raise NotImplementedError

    def to_task_definitions(self, cluster_config: Dict, hardware_config: Optional[Dict] = None) -> List:
        """Convert this component to multiple TaskDefinitions (for components that can expand)."""
        task_def = self.to_task_definition(cluster_config, hardware_config)
        return [task_def] if task_def else []

    def get_name(self) -> str:
        """Get a unique name for this component."""
        return type(self).__name__.lower()

    def _add_cross_group_env_vars(self, task_def, attributes: List[str]):
        """Add environment variables for cross-group access."""
        if self.het_group_index is not None:
            for attr in attributes:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    if attr == "port":
                        # For ports, export the value directly
                        task_def.environment[
                            f"{self.get_name().upper()}_{attr.upper()}_HET_GROUP_{self.het_group_index}"
                        ] = str(value)
                    elif attr == "host":
                        # For hosts, export the hostname
                        task_def.environment[
                            f"{self.get_name().upper()}_{attr.upper()}_HET_GROUP_{self.het_group_index}"
                        ] = "$(hostname)"


@dataclass
class Server(Component):
    """Declarative server component - cross-group compatible by default."""

    model: str
    server_type: str = "vllm"
    gpus: int = 8
    nodes: int = 1
    args: str = ""
    entrypoint: Optional[str] = None
    port: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        super().__init__()  # Initialize cross-group capabilities
        if self.port is None:
            self.port = get_free_port(strategy="random")
        if self.name is None:
            self.name = f"{self.server_type}_server"

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to ServerTask with cross-group capabilities."""
        # Apply hardware config overrides
        num_gpus = hardware_config.get("server_gpus", self.gpus) if hardware_config else self.gpus
        num_nodes = hardware_config.get("server_nodes", self.nodes) if hardware_config else self.nodes
        partition = hardware_config.get("partition") if hardware_config else None

        task_def = TaskFactory.create_server_task(
            name=self.name,
            server_type=self.server_type,
            container=cluster_config["containers"].get(self.server_type, cluster_config["containers"]["nemo"]),
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            model_path=self.model,
            server_port=self.port,
            server_args=self.args,
            server_entrypoint=self.entrypoint,
            partition=partition,
        )

        # Add cross-group environment variables
        self._add_cross_group_env_vars(task_def, ["host", "port"])

        return task_def

    def get_name(self) -> str:
        return self.name


@dataclass
class Sandbox(Component):
    """Declarative sandbox component - cross-group compatible by default."""

    port: Optional[int] = None
    name: str = "sandbox"

    def __post_init__(self):
        super().__init__()  # Initialize cross-group capabilities
        if self.port is None:
            self.port = get_free_port(strategy="random")

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to SandboxTask with cross-group capabilities."""
        task_def = TaskFactory.create_sandbox_task(
            name=self.name,
            container=cluster_config["containers"]["sandbox"],
            port=self.port,
        )

        # Add cross-group environment variables
        self._add_cross_group_env_vars(task_def, ["host", "port"])

        return task_def

    def get_name(self) -> str:
        return self.name


@dataclass
class GenerateTask(Component):
    """Declarative generation task that can reference other components - cross-group compatible."""

    input_source: str  # Can be file or directory
    output_dir: str
    server: Optional[Server] = None
    server_ref: Optional[RuntimeRef] = None  # For cross-group server references
    sandbox: Optional[Sandbox] = None
    sandbox_ref: Optional[RuntimeRef] = None  # For cross-group sandbox references
    extra_args: List[str] = field(default_factory=list)
    seeds: Optional[List[int]] = None  # If None, will be [None] for single job
    chunks: Optional[List[int]] = None  # If None, will be [None] for single job
    rerun_done: bool = False
    name: Optional[str] = None
    installation_command: Optional[str] = None

    def __post_init__(self):
        super().__init__()  # Initialize cross-group capabilities
        if self.name is None:
            self.name = "generate"
        # Normalize seeds and chunks
        if self.seeds is None:
            self.seeds = [None]
        if self.chunks is None:
            self.chunks = [None]

    def discover_jobs(self) -> List[Dict]:
        """Discover what actual jobs need to run based on existing outputs."""
        if self.rerun_done:
            # Return all combinations
            return [{"seed": seed, "chunk_id": chunk_id} for seed in self.seeds for chunk_id in self.chunks]

        incomplete = []
        for seed in self.seeds:
            for chunk_id in self.chunks:
                output_file = get_chunked_rs_filename(self.output_dir, seed, chunk_id)
                if not os.path.exists(f"{output_file}.done"):
                    incomplete.append({"seed": seed, "chunk_id": chunk_id})

        return incomplete

    def to_task_definitions(self, cluster_config: Dict, hardware_config: Optional[Dict] = None) -> List:
        """Convert to multiple MainTask definitions based on job discovery."""
        jobs_needed = self.discover_jobs()

        if not jobs_needed:
            LOG.info(f"GenerateTask '{self.name}': All jobs complete, nothing to run")
            return []

        LOG.info(f"GenerateTask '{self.name}': Found {len(jobs_needed)} jobs to run")

        task_definitions = []
        for job in jobs_needed:
            seed = job["seed"]
            chunk_id = job["chunk_id"]

            # Build job-specific command
            cmd = self._build_command_for_job(seed, chunk_id, cluster_config)

            # Create job-specific name
            job_name = self.name
            if seed is not None:
                job_name += f"_rs{seed}"
            if chunk_id is not None:
                job_name += f"_chunk{chunk_id}"

            # Apply hardware config if provided
            num_gpus = 0 if self.server else 1
            if hardware_config:
                num_gpus = hardware_config.get("num_gpus", num_gpus)

            task_def = TaskFactory.create_generation_task(
                name=job_name,
                cmd=cmd,
                container=cluster_config["containers"]["nemo-skills"],
                num_gpus=num_gpus,
                num_nodes=hardware_config.get("num_nodes", 1) if hardware_config else 1,
                partition=hardware_config.get("partition") if hardware_config else None,
                installation_command=self.installation_command,
            )

            # Apply additional hardware config
            if hardware_config:
                if hardware_config.get("exclusive"):
                    task_def.resources.slurm_kwargs = {"exclusive": True}
                if hardware_config.get("time_min"):
                    task_def.resources.time_min = hardware_config["time_min"]

            task_definitions.append(task_def)

        return task_definitions

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to MainTask - for backward compatibility."""
        task_defs = self.to_task_definitions(cluster_config, hardware_config)
        return task_defs[0] if task_defs else None

    def _build_command_for_job(
        self, seed: Optional[int], chunk_id: Optional[int], cluster_config: Optional[Dict] = None
    ) -> str:
        """Build generation command for a specific seed/chunk job."""
        # Determine input file for this job
        if os.path.isfile(self.input_source):
            input_file = self.input_source
        elif os.path.isdir(self.input_source) and seed is not None:
            input_file = f"{self.input_source}/output-rs{seed}.jsonl"
        else:
            input_file = self.input_source

        output_file = get_chunked_rs_filename(self.output_dir, seed, chunk_id)

        cmd_parts = [
            "export HYDRA_FULL_ERROR=1 &&",
            "python -m nemo_skills.inference.generate",
            "++skip_filled=True",
            f"++input_file={input_file}",
            f"++output_file={output_file}",
        ]

        # Add seed-specific parameters
        if seed is not None:
            cmd_parts.extend(
                [
                    f"++inference.random_seed={seed}",
                    "++inference.temperature=0.7",
                    "++inference.top_k=-1",
                    "++inference.top_p=0.95",
                ]
            )

        # Handle server references (local or cross-group)
        if self.server_ref:
            # Cross-group server reference - resolve at runtime
            server_host = self.server_ref.resolve_expression(cluster_config)
            server_port_ref = RuntimeRef(self.server_ref.component_name, self.server_ref.het_group_index, "port")
            server_port = server_port_ref.resolve_expression(cluster_config)

            # We need to determine server_type and model from the reference
            # For now, we'll need to pass these as additional attributes or handle differently
            cmd_parts.extend(
                [
                    f"++server.host={server_host}",
                    f"++server.port={server_port}",
                    # Note: server_type and model would need to be passed differently for cross-group refs
                ]
            )
        elif self.server:
            # Local server reference
            cmd_parts.extend(
                [
                    f"++server.server_type={self.server.server_type}",
                    "++server.host=127.0.0.1",
                    f"++server.port={self.server.port}",
                    f"++server.model={self.server.model}",
                ]
            )

        # Handle sandbox references (local or cross-group)
        if self.sandbox_ref:
            # Cross-group sandbox reference - resolve at runtime
            sandbox_host = self.sandbox_ref.resolve_expression(cluster_config)
            sandbox_port_ref = RuntimeRef(self.sandbox_ref.component_name, self.sandbox_ref.het_group_index, "port")
            sandbox_port = sandbox_port_ref.resolve_expression(cluster_config)

            cmd_parts.extend(
                [
                    f"++sandbox.host={sandbox_host}",
                    f"++sandbox.port={sandbox_port}",
                ]
            )
        elif self.sandbox:
            # Local sandbox reference
            cmd_parts.extend(
                [
                    "++sandbox.host=127.0.0.1",
                    f"++sandbox.port={self.sandbox.port}",
                ]
            )

        # Add extra arguments
        cmd_parts.extend(self.extra_args)

        # Add completion marker
        cmd_parts.extend(["&&", f"touch {output_file}.done"])

        return " ".join(cmd_parts)

    def get_name(self) -> str:
        return self.name


@dataclass
class TrainTask(Component):
    """Declarative training task."""

    training_data: str
    output_dir: str
    model_config: str
    gpus: int = 8
    nodes: int = 1
    name: Optional[str] = None

    def __post_init__(self):
        super().__init__()  # Initialize cross-group capabilities
        if self.name is None:
            self.name = "training"

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to MainTask."""
        cmd = f"python train.py --data {self.training_data} --output {self.output_dir} --config {self.model_config}"

        # Apply hardware config overrides
        num_gpus = hardware_config.get("num_gpus", self.gpus) if hardware_config else self.gpus
        num_nodes = hardware_config.get("num_nodes", self.nodes) if hardware_config else self.nodes
        partition = hardware_config.get("partition") if hardware_config else None
        exclusive = hardware_config.get("exclusive", False) if hardware_config else False

        return TaskFactory.create_training_task(
            name=self.name,
            cmd=cmd,
            container=cluster_config["containers"]["nemo"],
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
            exclusive=exclusive,
        )

    def get_name(self) -> str:
        return self.name


@dataclass
class RunCmd(Component):
    """Declarative component for running arbitrary bash commands in containers."""

    command: str
    container: str = "nemo-skills"
    gpus: Optional[int] = None
    nodes: int = 1
    name: Optional[str] = None
    working_dir: str = "/nemo_run/code"
    env_vars: Dict[str, str] = field(default_factory=dict)
    installation_command: Optional[str] = None

    def __post_init__(self):
        super().__init__()  # Initialize cross-group capabilities
        if self.name is None:
            self.name = "runcmd"

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to MainTask for running arbitrary commands."""
        # Build the command with proper environment setup
        cmd_parts = [
            "export HYDRA_FULL_ERROR=1",
            f"export PYTHONPATH=$PYTHONPATH:{self.working_dir}",
            f"cd {self.working_dir}",
        ]

        # Add any custom environment variables
        for env_var, value in self.env_vars.items():
            cmd_parts.append(f"export {env_var}={value}")

        # Add the actual command
        cmd_parts.append(self.command.strip())

        cmd = " && ".join(cmd_parts)

        # Apply hardware config overrides with proper None handling
        num_gpus = (
            hardware_config.get("num_gpus")
            if hardware_config and hardware_config.get("num_gpus") is not None
            else self.gpus
        )
        num_nodes = (
            hardware_config.get("num_nodes")
            if hardware_config and hardware_config.get("num_nodes") is not None
            else self.nodes
        )
        partition = hardware_config.get("partition") if hardware_config else None
        exclusive = hardware_config.get("exclusive", False) if hardware_config else False

        # Determine container from cluster config or use specified
        container_name = self.container
        if container_name in cluster_config.get("containers", {}):
            container_image = cluster_config["containers"][container_name]
        else:
            container_image = container_name  # Assume it's a direct container image name

        return TaskFactory.create_generation_task(  # Reuse generation task factory
            name=self.name,
            cmd=cmd,
            container=container_image,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
            installation_command=self.installation_command,
            exclusive=exclusive,
        )

    def get_name(self) -> str:
        return self.name


@dataclass
class HardwareConfig:
    """Hardware configuration for a group of tasks."""

    partition: Optional[str] = None
    time_min: Optional[str] = None
    exclusive: bool = False
    # Server-specific overrides
    server_gpus: Optional[int] = None
    server_nodes: Optional[int] = None
    # Generation task overrides
    num_gpus: Optional[int] = None
    num_nodes: Optional[int] = None


class HetGroup:
    """Heterogeneous group where components run with different resource requirements."""

    def __init__(self, components: List[Component], hardware: Optional[HardwareConfig] = None):
        self.components = components
        self.hardware = hardware or HardwareConfig()
        self.dependencies: List[Union["HetGroup", str]] = []
        self._name: Optional[str] = None

    def depends_on(self, dependency: Union["HetGroup", str]) -> "HetGroup":
        """Add dependency and return self for chaining."""
        self.dependencies.append(dependency)
        return self

    def named(self, name: str) -> "HetGroup":
        """Set explicit name and return self for chaining."""
        self._name = name
        return self

    def with_hardware(self, **kwargs) -> "HetGroup":
        """Set hardware configuration and return self for chaining."""
        for key, value in kwargs.items():
            setattr(self.hardware, key, value)
        return self

    def to_task_group(self, cluster_config: Dict) -> TaskGroup:
        """Convert to actual TaskGroup."""
        name = self._name or self._generate_name()
        group = TaskGroup(name, heterogeneous=len(self.components) > 1)

        # Convert hardware config to dict for passing to components
        hardware_dict = {
            "partition": self.hardware.partition,
            "time_min": self.hardware.time_min,
            "exclusive": self.hardware.exclusive,
            "server_gpus": self.hardware.server_gpus,
            "server_nodes": self.hardware.server_nodes,
            "num_gpus": self.hardware.num_gpus,
            "num_nodes": self.hardware.num_nodes,
        }

        for component in self.components:
            # Components that can expand (like GenerateTask) return multiple task definitions
            task_defs = component.to_task_definitions(cluster_config, hardware_dict)
            for task_def in task_defs:
                group.add_task(task_def)

        for dep in self.dependencies:
            group.add_dependency(dep)

        return group

    def _generate_name(self) -> str:
        """Generate name from component types."""
        return "_".join(comp.get_name() for comp in self.components)


class Pipeline:
    """Top-level pipeline that composes groups."""

    def __init__(self, name: str, groups: List[HetGroup], cluster: Optional[str] = None):
        self.name = name
        self.groups = groups
        self.cluster = cluster
        self._cluster_config: Optional[Dict] = None
        self._assign_het_group_indices()

    def _assign_het_group_indices(self):
        """Assign het_group_index to all components for cross-group references."""
        for group_idx, group in enumerate(self.groups):
            for component in group.components:
                # All components are now cross-group compatible
                component.het_group_index = group_idx
                LOG.debug(f"Assigned het_group_index {group_idx} to {component.get_name()}")

    def _get_cluster_config(self) -> Dict:
        """Get cluster configuration, loading it if necessary."""
        if self._cluster_config is None:
            if self.cluster is None:
                raise ValueError("Must specify cluster either in Pipeline() or run() method")
            from nemo_skills.pipeline.utils import get_cluster_config

            self._cluster_config = get_cluster_config(self.cluster)
        return self._cluster_config

    def run(self, cluster_config: Optional[Dict] = None, cluster: Optional[str] = None, dry_run: bool = False):
        """Execute the pipeline."""
        if not self.groups:
            LOG.info("No groups to execute")
            return None

        # Determine cluster config to use
        if cluster_config is not None:
            # Explicit config provided
            final_cluster_config = cluster_config
        elif cluster is not None:
            # Cluster name provided
            from nemo_skills.pipeline.utils import get_cluster_config

            final_cluster_config = get_cluster_config(cluster)
        else:
            # Use pipeline's cluster
            final_cluster_config = self._get_cluster_config()

        builder = PipelineBuilder(self.name, final_cluster_config)

        # Convert all groups to task groups
        for group in self.groups:
            task_group = group.to_task_group(final_cluster_config)
            builder.add_task_group(task_group)

        # Execute
        with get_exp(self.name, final_cluster_config) as exp:
            builder.build_experiment(exp, "/tmp/logs")
            if not dry_run:
                run_exp(exp, final_cluster_config)
            return exp


class JobDiscovery:
    """Smart job discovery that determines what actually needs to run."""

    @staticmethod
    def find_incomplete_jobs(
        input_source: str,
        output_dir: str,
        seeds: List[Optional[int]],
        chunks: List[Optional[int]],
        rerun_done: bool = False,
    ) -> List[Dict]:
        """Find jobs that haven't been completed yet."""
        if rerun_done:
            # Return all combinations
            return [
                {"seed": seed, "chunk_id": chunk_id, "input_file": JobDiscovery._get_input_file(input_source, seed)}
                for seed in seeds
                for chunk_id in chunks
            ]

        incomplete = []
        for seed in seeds:
            for chunk_id in chunks:
                output_file = get_chunked_rs_filename(output_dir, seed, chunk_id)
                if not os.path.exists(f"{output_file}.done"):
                    incomplete.append(
                        {
                            "seed": seed,
                            "chunk_id": chunk_id,
                            "input_file": JobDiscovery._get_input_file(input_source, seed),
                        }
                    )

        return incomplete

    @staticmethod
    def _get_input_file(input_source: str, seed: Optional[int]) -> str:
        """Get input file for a specific seed."""
        if os.path.isfile(input_source):
            return input_source
        elif os.path.isdir(input_source) and seed is not None:
            return f"{input_source}/output-rs{seed}.jsonl"
        else:
            raise ValueError(f"Cannot determine input file from {input_source} with seed {seed}")


class GenerationPipeline:
    """High-level generation pipeline with smart defaults."""

    @staticmethod
    def auto_create(
        input_source: str,
        output_dir: str,
        model: str,
        server_type: str = "vllm",
        server_gpus: int = 2,
        server_nodes: int = 1,
        server_args: str = "",
        with_sandbox: bool = False,
        seeds: Optional[List[int]] = None,
        chunks: Optional[List[int]] = None,
        dependent_jobs: int = 0,
        extra_args: List[str] = None,
        rerun_done: bool = False,
        cluster_config: Dict = None,
    ) -> Pipeline:
        """Auto-create pipeline by discovering what jobs need to run."""

        # Auto-discover incomplete jobs
        jobs_needed = JobDiscovery.find_incomplete_jobs(
            input_source=input_source,
            output_dir=output_dir,
            seeds=seeds or [None],
            chunks=chunks or [None],
            rerun_done=rerun_done,
        )

        if not jobs_needed:
            LOG.info("All jobs complete")
            return Pipeline("empty", [])

        LOG.info(f"Found {len(jobs_needed)} jobs that need to run")

        # Create shared components (reused across jobs)
        shared_server = None
        shared_sandbox = None

        if server_gpus and server_gpus > 0:
            shared_server = Server(
                model=model,
                server_type=server_type,
                gpus=server_gpus,
                nodes=server_nodes,
                args=server_args,
            )

        if with_sandbox:
            shared_sandbox = Sandbox()

        # Create groups for each job + dependent jobs
        groups = []
        for job in jobs_needed:
            for dep_job_idx in range(dependent_jobs + 1):
                components = []

                # Add shared server if needed
                if shared_server:
                    components.append(shared_server)

                # Add shared sandbox if needed
                if shared_sandbox:
                    components.append(shared_sandbox)

                # Create generation task
                job_name = f"generate_rs{job['seed']}" if job["seed"] is not None else "generate"
                if job["chunk_id"] is not None:
                    job_name += f"_chunk{job['chunk_id']}"
                if dep_job_idx > 0:
                    job_name += f"_job{dep_job_idx}"

                generation = GenerateTask(
                    input_file=job["input_file"],
                    output_dir=output_dir,
                    server=shared_server,
                    sandbox=shared_sandbox,
                    seed=job["seed"],
                    chunk_id=job["chunk_id"],
                    extra_args=extra_args or [],
                    name=job_name,
                )
                components.append(generation)

                # Create heterogeneous group
                group = HetGroup(components).named(job_name)
                groups.append(group)

        return Pipeline("generation", groups)
