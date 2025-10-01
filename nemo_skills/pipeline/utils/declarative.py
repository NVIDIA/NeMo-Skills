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
Simplified declarative pipeline system using only Command for all task types.

Example usage:
    from nemo_skills.pipeline.utils.commands import vllm_server_command, sandbox_command

    server = Command(command=vllm_server_command(model="Qwen/Qwen3-8B"), gpus=8, name="server")
    sandbox = Command(command=sandbox_command(), name="sandbox")
    client = Command(
        command=lambda: f"curl {server.hostname_ref()}:{server.meta_ref('port')}/health",
        name="client"
    )

    pipeline = Pipeline(
        name="my_pipeline",
        cluster="local",
        groups=[HetGroup([server, sandbox, client])]
    )
    pipeline.run()
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from nemo_skills.pipeline.utils import get_exp, run_exp
from nemo_skills.pipeline.utils.task_factories import TaskFactory
from nemo_skills.pipeline.utils.task_system import PipelineBuilder, TaskGroup
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def reference_property(attribute_name: str):
    """Decorator that creates reference methods for component attributes."""

    def decorator(cls):
        def make_var_method(attr_name):
            def var_method(self) -> str:
                """Get environment variable name for this attribute."""
                return f"{self.get_name().upper()}_{attr_name.upper()}_HET_GROUP_{self.het_group_index}"

            return var_method

        def make_ref_method(attr_name):
            def ref_method(self) -> str:
                """Get shell reference for this attribute."""
                if self.het_group_index is None:
                    # Fallback for local execution
                    if attr_name == "host":
                        return "127.0.0.1"
                    elif attr_name == "port" and hasattr(self, "port"):
                        return str(self.port)
                    else:
                        return "localhost"  # Safe fallback

                # For heterogeneous jobs
                if attr_name == "host":
                    # SLURM_JOB_NODELIST_HET_GROUP_N contains the hostname(s) for this het component
                    # Just extract the first one if there are multiple (comma-separated)
                    # For a single node, it's already the full hostname (e.g., "cw-dfw-h100-004-004-033")
                    return f"$(echo $SLURM_JOB_NODELIST_HET_GROUP_{self.het_group_index} | cut -d',' -f1)"
                else:
                    # For other attributes (like port), use environment variables
                    var_name = f"{self.get_name().upper()}_{attr_name.upper()}_HET_GROUP_{self.het_group_index}"
                    return f"${{{var_name}}}"

            return ref_method

        # Add var and ref methods for the attribute
        setattr(cls, f"{attribute_name}_var", make_var_method(attribute_name))
        setattr(cls, f"{attribute_name}_ref", make_ref_method(attribute_name))

        # Add to reference registry for discovery
        if not hasattr(cls, "_reference_attributes"):
            cls._reference_attributes = []
        cls._reference_attributes.append(attribute_name)

        return cls

    return decorator


def reference_component(cls):
    """Class decorator that adds discovery methods to components."""

    def discover_vars(self) -> Dict[str, str]:
        """Discover all environment variable names this component will create."""
        vars_dict = {}
        for attr in getattr(self, "_reference_attributes", []):
            var_method = getattr(self, f"{attr}_var")
            vars_dict[f"{attr}_var"] = var_method()
        return vars_dict

    def discover_refs(self) -> Dict[str, str]:
        """Discover all shell references this component provides."""
        refs_dict = {}
        for attr in getattr(self, "_reference_attributes", []):
            ref_method = getattr(self, f"{attr}_ref")
            refs_dict[f"{attr}_ref"] = ref_method()
        return refs_dict

    # Add discovery methods
    cls.discover_vars = discover_vars
    cls.discover_refs = discover_refs

    return cls


@dataclass
class RuntimeRef:
    """Reference to a component that will be resolved at runtime using SLURM environment variables."""

    component_name: str
    het_group_index: int
    attribute: str = "host"  # What to resolve (host, port, etc.)

    def resolve_expression(self) -> str:
        """Get the runtime expression to resolve this reference."""
        if self.attribute == "host":
            # Use SLURM environment variable to get first host in the het group
            return f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{self.het_group_index} | head -n1)"
        elif self.attribute == "port":
            # Use environment variable set by the component
            return f"${{{self.component_name.upper()}_PORT_HET_GROUP_{self.het_group_index}}}"
        else:
            return f"${{{self.component_name.upper()}_{self.attribute.upper()}_HET_GROUP_{self.het_group_index}}}"

    def __str__(self):
        return f"RuntimeRef({self.component_name}.{self.attribute} from group {self.het_group_index})"


# Component base functionality is now merged into Command class below
# GenerateTask and TrainTask have been removed - use Command with command builders instead


@reference_component
@reference_property("host")
@reference_property("port")
@dataclass
class Command:
    """Declarative command for running tasks in containers.

    The command can be either:
    - A string: evaluated immediately
    - A callable (lambda): evaluated lazily when the task is prepared
    - A tuple (command, metadata): command with metadata like port
    - A callable returning (command, metadata): lazy evaluation with metadata

    Using a lambda allows references to work correctly in heterogeneous jobs:
        Command(command=lambda: f"curl {server.hostname_ref()}:5000")

    Metadata from command builders (like port) can be referenced:
        server = Command(command=vllm_server_command(...))
        client = Command(command=lambda: f"curl {server.hostname_ref()}:{server.meta_ref('port')}")
    """

    command: Union[str, Callable[[], str], Tuple[str, Dict], Callable[[], Tuple[str, Dict]]]
    container: str = "nemo-skills"
    gpus: Optional[int] = None
    nodes: int = 1
    name: Optional[str] = None
    working_dir: str = "/nemo_run/code"
    env_vars: Dict[str, str] = field(default_factory=dict)
    installation_command: Optional[str] = None
    port: Optional[int] = None  # Can be set from metadata
    metadata: Dict[str, any] = field(default_factory=dict)  # Stores metadata from command builders
    het_group_index: Optional[int] = None  # Set automatically by Pipeline

    def __post_init__(self):
        # Initialize component (merged from old Component class)
        if self.name is None:
            self.name = "command"

        # Extract metadata if command is a tuple or callable returning tuple
        self._extract_metadata()

    def _extract_metadata(self):
        """Extract metadata from command if it's a tuple."""
        if not callable(self.command):
            if isinstance(self.command, tuple):
                self.command, self.metadata = self.command
                # Set port if it's in metadata
                if "port" in self.metadata:
                    self.port = self.metadata["port"]

    def hostname_ref(self) -> str:
        """Get hostname reference for this component (for cross-group access)."""
        return self.host_ref()

    def meta_ref(self, key: str) -> str:
        """Get metadata value reference (like port)."""
        if key == "port" and self.port is not None:
            return self.port_ref()
        elif key in self.metadata:
            # For non-port metadata, return the value directly (as string)
            return str(self.metadata[key])
        else:
            return ""

    def to_task_definition(self, cluster_config: Dict, hardware_config: Optional[Dict] = None):
        """Convert to MainTask/ServerTask/SandboxTask based on metadata."""
        # Evaluate command if it's a callable (lazy evaluation)
        if callable(self.command):
            result = self.command()
            if isinstance(result, tuple):
                actual_command, self.metadata = result
                if "port" in self.metadata and self.port is None:
                    self.port = self.metadata["port"]
            else:
                actual_command = result
        else:
            actual_command = self.command

        # Check if this is a server or sandbox based on metadata
        is_server = "server_type" in self.metadata
        is_sandbox = "port" in self.metadata and not is_server and actual_command.startswith("# Sandbox")

        # Apply hardware config overrides
        num_gpus = self.gpus if self.gpus is not None else 0
        num_nodes = hardware_config.get("num_nodes", self.nodes) if hardware_config else self.nodes
        partition = hardware_config.get("partition") if hardware_config else None
        exclusive = hardware_config.get("exclusive", False) if hardware_config else False

        if is_server:
            # This is a server - use ServerTask

            server_config = {
                "server_type": self.metadata["server_type"],
                "num_gpus": self.metadata.get("gpus", num_gpus),
                "num_nodes": self.metadata.get("nodes", num_nodes),
                "model_path": self.metadata["model"],
                "server_port": self.metadata["port"],
                "server_args": self.metadata.get("server_args", ""),
                "cluster_config": cluster_config,
            }
            if self.metadata.get("server_entrypoint"):
                server_config["server_entrypoint"] = self.metadata["server_entrypoint"]

            # Determine container
            container_name = cluster_config["containers"].get(
                self.metadata["server_type"],
                cluster_config["containers"].get("nemo", cluster_config["containers"].get("nemo-skills")),
            )

            task_def = TaskFactory.create_server_task(
                name=self.name,
                server_type=self.metadata["server_type"],
                container=container_name,
                num_gpus=self.metadata.get("gpus", num_gpus),
                num_nodes=self.metadata.get("nodes", num_nodes),
                model_path=self.metadata["model"],
                server_port=self.metadata["port"],
                server_args=self.metadata.get("server_args", ""),
                server_entrypoint=self.metadata.get("server_entrypoint"),
                partition=partition,
            )

            # Add cross-group environment variables
            self._add_cross_group_env_vars(task_def, ["host", "port"])

            return task_def

        elif is_sandbox:
            # This is a sandbox - use SandboxTask
            task_def = TaskFactory.create_sandbox_task(
                name=self.name,
                container=cluster_config["containers"]["sandbox"],
                port=self.metadata["port"],
            )

            # Add cross-group environment variables
            self._add_cross_group_env_vars(task_def, ["host", "port"])

            return task_def

        else:
            # Regular command - build with environment setup
            cmd_parts = [
                "export HYDRA_FULL_ERROR=1",
                f"export PYTHONPATH=$PYTHONPATH:{self.working_dir}",
                f"cd {self.working_dir}",
            ]

            # Add any custom environment variables
            for env_var, value in self.env_vars.items():
                cmd_parts.append(f"export {env_var}={value}")

            # Add the actual command
            cmd_parts.append(actual_command.strip())

            cmd = " && ".join(cmd_parts)

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

    def to_task_definitions(self, cluster_config: Dict, hardware_config: Optional[Dict] = None) -> List:
        """Convert this component to multiple TaskDefinitions (for components that can expand).

        Most commands return a single task definition.
        """
        task_def = self.to_task_definition(cluster_config, hardware_config)
        return [task_def] if task_def else []

    def _add_cross_group_env_vars(self, task_def, attributes: List[str]):
        """Add environment variables for cross-group access (currently only ports).

        Note: Hostnames are resolved directly from SLURM_JOB_NODELIST_HET_GROUP_N
        in the shell script, not through environment variables.
        """
        if self.het_group_index is not None:
            for attr in attributes:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    if attr == "port":
                        # For ports, export the value directly
                        task_def.environment[
                            f"{self.get_name().upper()}_{attr.upper()}_HET_GROUP_{self.het_group_index}"
                        ] = str(value)


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
    """Heterogeneous group where commands run with different resource requirements."""

    def __init__(self, commands: List[Command], hardware: Optional[HardwareConfig] = None):
        self.components = commands  # Keep as self.components internally for backwards compatibility
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

    def to_task_group(self, cluster_config: Dict, pipeline_output_dir: Optional[str] = None) -> TaskGroup:
        """Convert to actual TaskGroup."""
        name = self._name or self._generate_name()
        # All components in a HetGroup share the same resources, so NOT heterogeneous within the group
        group = TaskGroup(name, heterogeneous=False)

        # Convert hardware config to dict for passing to components
        hardware_dict = {
            "partition": self.hardware.partition,
            "time_min": self.hardware.time_min,
            "exclusive": self.hardware.exclusive,
            "server_gpus": self.hardware.server_gpus,
            "server_nodes": self.hardware.server_nodes,
            "num_gpus": self.hardware.num_gpus,
            "num_nodes": self.hardware.num_nodes,
            "pipeline_output_dir": pipeline_output_dir,  # ✨ Pass pipeline output dir
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

    def __init__(
        self, name: str, groups: List[HetGroup], cluster: Optional[str] = None, output_dir: Optional[str] = None
    ):
        self.name = name
        self.groups = groups
        self.cluster = cluster
        self.output_dir = output_dir or f"/tmp/{name}_outputs"
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

        # If multiple groups, they need special handling for heterogeneous jobs
        if len(self.groups) > 1:
            # Each HetGroup becomes a separate het component
            # Update het_group_indices for components to match their actual het component positions
            for het_idx, group in enumerate(self.groups):
                # Update all components in this group to have the correct het_group_index
                for component in group.components:
                    component.het_group_index = het_idx

                task_group = group.to_task_group(final_cluster_config, self.output_dir)
                # Mark that this is part of a multi-group pipeline
                task_group._is_het_component = True
                builder.add_task_group(task_group)
        else:
            # Single group - just add it directly
            task_group = self.groups[0].to_task_group(final_cluster_config, self.output_dir)
            builder.add_task_group(task_group)

        # Execute
        log_dir = f"{self.output_dir}/logs"
        with get_exp(self.name, final_cluster_config) as exp:
            builder.build_experiment(exp, log_dir)
            if not dry_run:
                run_exp(exp, final_cluster_config)
            return exp


# JobDiscovery and GenerationPipeline have been removed
# Use Command with command builders to create custom pipelines instead
