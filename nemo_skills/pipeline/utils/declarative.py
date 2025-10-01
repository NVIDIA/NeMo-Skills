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
from nemo_skills.pipeline.utils.commands import wrap_command
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

    command: Union[
        str,
        Callable[[], str],  # Lambda for cross-group refs
        Callable[[Dict], str],  # Lambda needing cluster_config (e.g., sandbox)
        Tuple[Optional[str], Dict],  # Command builder result
        Callable[[], Tuple[Optional[str], Dict]],  # Lambda returning command builder result
        None,
    ]
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

        # Extract metadata if command is a tuple
        self._extract_metadata()

        # Wrap plain strings with environment setup
        if isinstance(self.command, str) and (self.env_vars or self.working_dir):
            self.command = wrap_command(self.command, self.working_dir, self.env_vars)

    def _extract_metadata(self):
        """Extract metadata from command if it's a tuple."""
        if not callable(self.command):
            if isinstance(self.command, tuple):
                cmd, self.metadata = self.command
                self.command = cmd  # Can be None for sandbox
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

    def prepare_for_execution(self, cluster_config: Dict) -> Tuple[str, Dict]:
        """Prepare command for execution.

        This method:
        1. Evaluates callables (resolves cross-group references and cluster_config-dependent commands)
        2. Adds cross-group environment variables

        Returns:
            Tuple of (final_command, execution_config)
        """
        # 1. Evaluate if callable (resolves cross-group references or cluster_config needs)
        if callable(self.command):
            # Check if lambda needs cluster_config (for sandbox)
            import inspect

            sig = inspect.signature(self.command)
            if len(sig.parameters) > 0:
                # Lambda expects cluster_config
                result = self.command(cluster_config)
            else:
                # Regular lambda (cross-group refs)
                result = self.command()

            if isinstance(result, tuple):
                final_command, runtime_metadata = result
                # Deep merge metadata, especially environment dict
                for key, value in runtime_metadata.items():
                    if key == "environment" and key in self.metadata:
                        # Merge environment dicts instead of replacing
                        self.metadata[key].update(value)
                    else:
                        self.metadata[key] = value
            else:
                final_command = result
        else:
            final_command = self.command

        # 2. Add cross-group environment variables (for cross-group references)
        if self.het_group_index is not None and self.port is not None:
            env_vars = self.metadata.setdefault("environment", {})
            env_vars[f"{self.name.upper()}_PORT_HET_GROUP_{self.het_group_index}"] = str(self.port)

        # 3. Build execution config from metadata
        execution_config = {
            "num_tasks": self.metadata.get("num_tasks", 1),
            "num_gpus": self.metadata.get("gpus", self.gpus or 0),
            "num_nodes": self.metadata.get("nodes", self.nodes),
            "environment": self.metadata.get("environment", {}),
            "log_prefix": self.metadata.get("log_prefix", "main"),
            "mounts": self.metadata.get("mounts"),
            "container": self.metadata.get("container", self.container),  # Use container from metadata if available
        }

        return final_command, execution_config

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
        """Execute the pipeline by calling NeMo-Run directly."""
        if not self.groups:
            LOG.info("No groups to execute")
            return None

        # Determine cluster config to use
        if cluster_config is not None:
            final_cluster_config = cluster_config
        elif cluster is not None:
            from nemo_skills.pipeline.utils import get_cluster_config

            final_cluster_config = get_cluster_config(cluster)
        else:
            final_cluster_config = self._get_cluster_config()

        # Assign het_group_indices for cross-group references
        for het_idx, group in enumerate(self.groups):
            for command in group.components:
                command.het_group_index = het_idx

        log_dir = f"{self.output_dir}/logs"

        with get_exp(self.name, final_cluster_config) as exp:
            if len(self.groups) > 1:
                # Multiple HetGroups = heterogeneous job
                self._add_heterogeneous_job(exp, final_cluster_config, log_dir)
            else:
                # Single group
                self._add_group(exp, self.groups[0], final_cluster_config, log_dir)

            if not dry_run:
                run_exp(exp, final_cluster_config)
            return exp

    def _add_group(self, exp, group: HetGroup, cluster_config: Dict, log_dir: str):
        """Add a single HetGroup to the experiment."""
        import nemo_run as run

        from nemo_skills.pipeline.utils import get_executor, temporary_env_update

        commands = []
        executors = []

        for command in group.components:
            # Command prepares itself!
            final_cmd, exec_config = command.prepare_for_execution(cluster_config)
            commands.append(final_cmd)

            # Determine container (use from exec_config if available, e.g., for sandbox)
            container_name = exec_config.get("container", command.container)
            if container_name in cluster_config.get("containers", {}):
                container_image = cluster_config["containers"][container_name]
            else:
                container_image = container_name

            # Build executor
            if exec_config.get("environment"):
                with temporary_env_update(cluster_config, exec_config["environment"]):
                    executor = get_executor(
                        cluster_config=cluster_config,
                        container=container_image,
                        num_nodes=exec_config["num_nodes"],
                        tasks_per_node=exec_config["num_tasks"],
                        gpus_per_node=exec_config["num_gpus"],
                        job_name=command.name,
                        log_dir=log_dir,
                        log_prefix=exec_config["log_prefix"],
                        partition=group.hardware.partition if group.hardware else None,
                        mounts=exec_config.get("mounts"),
                    )
            else:
                executor = get_executor(
                    cluster_config=cluster_config,
                    container=container_image,
                    num_nodes=exec_config["num_nodes"],
                    tasks_per_node=exec_config["num_tasks"],
                    gpus_per_node=exec_config["num_gpus"],
                    job_name=command.name,
                    log_dir=log_dir,
                    log_prefix=exec_config["log_prefix"],
                    partition=group.hardware.partition if group.hardware else None,
                )

            executors.append(executor)

        # Add to experiment
        if len(commands) == 1:
            exp.add(
                run.Script(inline=commands[0]),
                executor=executors[0],
                name="nemo-run",
            )
        else:
            exp.add(
                [run.Script(inline=cmd) for cmd in commands],
                executor=executors,
                name="nemo-run",
            )

    def _add_heterogeneous_job(self, exp, cluster_config: Dict, log_dir: str):
        """Add multiple HetGroups as a single heterogeneous SLURM job."""
        import nemo_run as run

        from nemo_skills.pipeline.utils import get_executor, temporary_env_update

        LOG.info(f"Creating heterogeneous job with {len(self.groups)} HetGroup components")

        all_commands = []
        all_executors = []
        het_group_indices = []

        # Collect environment variables from all commands (for cross-component refs)
        shared_env_vars = {}
        for het_idx, group in enumerate(self.groups):
            for command in group.components:
                _, exec_config = command.prepare_for_execution(cluster_config)
                shared_env_vars.update(exec_config.get("environment", {}))

        # Build commands and executors
        for het_idx, group in enumerate(self.groups):
            LOG.info(f"Het component {het_idx}: {len(group.components)} commands from group '{group._name}'")

            for command in group.components:
                # Prepare command
                final_cmd, exec_config = command.prepare_for_execution(cluster_config)
                all_commands.append(final_cmd)
                het_group_indices.append(het_idx)

                # Merge shared environment
                exec_config["environment"].update(shared_env_vars)

                # Determine container (use from exec_config if available, e.g., for sandbox)
                container_name = exec_config.get("container", command.container)
                if container_name in cluster_config.get("containers", {}):
                    container_image = cluster_config["containers"][container_name]
                else:
                    container_image = container_name

                # Build executor
                if exec_config.get("environment"):
                    with temporary_env_update(cluster_config, exec_config["environment"]):
                        executor = get_executor(
                            cluster_config=cluster_config,
                            container=container_image,
                            num_nodes=exec_config["num_nodes"],
                            tasks_per_node=exec_config["num_tasks"],
                            gpus_per_node=exec_config["num_gpus"],
                            job_name=command.name,
                            log_dir=log_dir,
                            log_prefix=exec_config["log_prefix"],
                            partition=group.hardware.partition if group.hardware else None,
                            heterogeneous=True,
                            het_group=het_idx,
                            total_het_groups=len(self.groups),
                            overlap=len(group.components) > 1,
                            mounts=exec_config.get("mounts"),
                        )
                else:
                    executor = get_executor(
                        cluster_config=cluster_config,
                        container=container_image,
                        num_nodes=exec_config["num_nodes"],
                        tasks_per_node=exec_config["num_tasks"],
                        gpus_per_node=exec_config["num_gpus"],
                        job_name=command.name,
                        log_dir=log_dir,
                        log_prefix=exec_config["log_prefix"],
                        partition=group.hardware.partition if group.hardware else None,
                        heterogeneous=True,
                        het_group=het_idx,
                        total_het_groups=len(self.groups),
                        overlap=len(group.components) > 1,
                    )

                all_executors.append(executor)

        # Set het_group_indices on first executor
        all_executors[0].het_group_indices = het_group_indices

        # Add to experiment
        exp.add(
            [run.Script(inline=cmd) for cmd in all_commands],
            executor=all_executors,
            name="nemo-run",
        )
