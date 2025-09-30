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

import logging
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import nemo_run as run

from nemo_skills.pipeline.utils.cluster import temporary_env_update
from nemo_skills.pipeline.utils.exp import get_executor, install_packages_wrap
from nemo_skills.pipeline.utils.server import get_server_command
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class ResourceSpec:
    """Specification for computational resources needed by a task."""

    num_tasks: int = 1
    num_gpus: Optional[int] = None
    num_nodes: int = 1
    partition: Optional[str] = None
    time_min: Optional[str] = None
    memory: Optional[str] = None
    exclusive: bool = False
    slurm_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.slurm_kwargs is None:
            self.slurm_kwargs = {}


@dataclass
class TaskDefinition(ABC):
    """Base class for all task definitions."""

    name: str
    command: Union[str, List[str]]
    container: str
    resources: ResourceSpec
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    log_prefix: str = "main"

    @abstractmethod
    def prepare_command(self, cluster_config: Dict[str, Any], **kwargs) -> str:
        """Prepare the final command to be executed."""
        pass

    def get_executor_kwargs(self) -> Dict[str, Any]:
        """Get additional kwargs for executor creation."""
        return {}


@dataclass
class MainTask(TaskDefinition):
    """Main computational task (training, generation, evaluation, etc.)."""

    installation_command: Optional[str] = None

    def prepare_command(self, cluster_config: Dict[str, Any], **kwargs) -> str:
        cmd = self.command
        if isinstance(cmd, list):
            cmd = " && ".join(cmd)

        # Apply installation wrapper if needed
        if self.installation_command:
            cmd = install_packages_wrap(cmd, self.installation_command)

        return cmd


@dataclass
class ServerTask(TaskDefinition):
    """Server task for hosting models or services."""

    server_type: str = ""
    server_args: str = ""
    server_entrypoint: Optional[str] = None
    model_path: Optional[str] = None
    server_port: Optional[int] = None

    def prepare_command(self, cluster_config: Dict[str, Any], **kwargs) -> str:
        # Get or generate server port
        server_port = self.server_port
        if server_port is None:
            from nemo_skills.pipeline.utils.server import get_free_port

            server_port = get_free_port(strategy="random")

        # Build server configuration
        server_config = {
            "server_type": self.server_type,
            "num_gpus": self.resources.num_gpus or 0,
            "num_nodes": self.resources.num_nodes or 1,  # Default to 1 if None
            "model_path": self.model_path or "",
            "server_port": server_port,
            "server_args": self.server_args,
            "cluster_config": cluster_config,
        }

        if self.server_entrypoint:
            server_config["server_entrypoint"] = self.server_entrypoint

        cmd, num_tasks = get_server_command(**server_config)

        # Update resources with actual task count from server
        self.resources.num_tasks = num_tasks

        # Handle MPI for non-slurm executors
        if cluster_config["executor"] != "slurm" and num_tasks > 1:
            cmd = f"mpirun --allow-run-as-root -np {num_tasks} bash -c {shlex.quote(cmd)}"

        return cmd


@dataclass
class SandboxTask(TaskDefinition):
    """Sandbox task for code execution environments."""

    port: Optional[int] = None

    def prepare_command(self, cluster_config: Dict[str, Any], **kwargs) -> str:
        # Set up environment variables for sandbox
        if self.port is not None:
            self.environment.update(
                {
                    "LISTEN_PORT": str(self.port),
                    "NGINX_PORT": str(self.port),
                }
            )

            # Handle PYTHONPATH for sandbox
            current_env_vars = cluster_config.get("env_vars", [])
            for env_var in current_env_vars:
                if "PYTHONPATH" in env_var:
                    if env_var.startswith("PYTHONPATH="):
                        pythonpath = env_var[11:]
                    else:
                        pythonpath = env_var
                    self.environment["PYTHONPATH"] = pythonpath + ":/app"
                    break

        if cluster_config["executor"] == "none":
            return "python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server"
        return "/start-with-nginx.sh"

    def get_executor_kwargs(self) -> Dict[str, Any]:
        return {"mounts": []}  # Sandbox doesn't mount anything


class TaskGroup:
    """Group of related tasks that execute together, potentially heterogeneously."""

    def __init__(self, name: str, heterogeneous: bool = False):
        self.name = name
        self.tasks: List[TaskDefinition] = []
        self.dependencies: List[Union["TaskGroup", str]] = []
        self.heterogeneous = heterogeneous
        self._task_id: Optional[str] = None

    def add_task(self, task: TaskDefinition) -> TaskDefinition:
        """Add a task to this group."""
        self.tasks.append(task)
        return task

    def add_dependency(self, dependency: Union["TaskGroup", str]):
        """Add a dependency on another task group or experiment."""
        self.dependencies.append(dependency)

    def get_task_id(self) -> Optional[str]:
        """Get the task ID after the group has been added to an experiment."""
        return self._task_id


class ExecutorBuilder:
    """Builds executors for tasks based on cluster configuration."""

    @staticmethod
    def build_executor(
        task: TaskDefinition,
        cluster_config: Dict[str, Any],
        log_dir: str,
        dependencies: Optional[List[str]] = None,
        het_group: int = 0,
        total_het_groups: int = 1,
        overlap: bool = False,
        with_ray: bool = False,
        extra_package_dirs: Optional[List[str]] = None,
    ):
        """Build an executor for a specific task."""

        # Get task-specific executor kwargs
        executor_kwargs = task.get_executor_kwargs()

        # Use temporary environment update if task has environment variables
        if task.environment:
            with temporary_env_update(cluster_config, task.environment):
                return get_executor(
                    cluster_config=cluster_config,
                    container=task.container,
                    num_nodes=task.resources.num_nodes or 1,  # Default to 1 if None
                    tasks_per_node=task.resources.num_tasks or 1,  # Default to 1 if None
                    gpus_per_node=task.resources.num_gpus,
                    job_name=task.name,
                    log_dir=log_dir,
                    log_prefix=task.log_prefix,
                    partition=task.resources.partition,
                    time_min=task.resources.time_min,
                    dependencies=dependencies,
                    extra_package_dirs=tuple(extra_package_dirs) if extra_package_dirs else None,
                    heterogeneous=len([t for t in [task] if isinstance(t, TaskDefinition)]) > 1,
                    het_group=het_group,
                    total_het_groups=total_het_groups,
                    slurm_kwargs=task.resources.slurm_kwargs,
                    overlap=overlap,
                    with_ray=with_ray,
                    **executor_kwargs,
                )
        else:
            return get_executor(
                cluster_config=cluster_config,
                container=task.container,
                num_nodes=task.resources.num_nodes or 1,  # Default to 1 if None
                tasks_per_node=task.resources.num_tasks or 1,  # Default to 1 if None
                gpus_per_node=task.resources.num_gpus,
                job_name=task.name,
                log_dir=log_dir,
                log_prefix=task.log_prefix,
                partition=task.resources.partition,
                time_min=task.resources.time_min,
                dependencies=dependencies,
                extra_package_dirs=tuple(extra_package_dirs) if extra_package_dirs else None,
                heterogeneous=len([t for t in [task] if isinstance(t, TaskDefinition)]) > 1,
                het_group=het_group,
                total_het_groups=total_het_groups,
                slurm_kwargs=task.resources.slurm_kwargs,
                overlap=overlap,
                with_ray=with_ray,
                **executor_kwargs,
            )


class PipelineBuilder:
    """Builder for creating complex task pipelines."""

    def __init__(self, experiment_name: str, cluster_config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.cluster_config = cluster_config
        self.task_groups: List[TaskGroup] = []
        self.global_config = {
            "reuse_code": True,
            "reuse_code_exp": None,
            "extra_package_dirs": None,
            "with_ray": False,
        }

    def set_global_config(self, **kwargs):
        """Set global configuration options."""
        self.global_config.update(kwargs)

    def add_task_group(self, task_group: TaskGroup) -> TaskGroup:
        """Add a task group to the pipeline."""
        self.task_groups.append(task_group)
        return task_group

    def build_experiment(self, exp: run.Experiment, log_dir: str) -> List[str]:
        """Build the experiment by adding all task groups."""
        task_ids = []

        for group in self.task_groups:
            task_id = self._add_task_group_to_experiment(exp, group, log_dir)
            group._task_id = task_id
            task_ids.append(task_id)

        return task_ids

    def _add_task_group_to_experiment(self, exp: run.Experiment, group: TaskGroup, log_dir: str) -> str:
        """Add a single task group to the experiment."""

        # Resolve dependencies
        dependencies = self._resolve_dependencies(group)

        # Prepare commands and executors
        commands = []
        executors = []
        het_group_indices = []

        total_het_groups = len(group.tasks) if group.heterogeneous else 1

        for i, task in enumerate(group.tasks):
            # Prepare the command
            cmd = task.prepare_command(self.cluster_config)
            commands.append(cmd)

            # Build executor
            het_group = i if group.heterogeneous else 0
            executor = ExecutorBuilder.build_executor(
                task=task,
                cluster_config=self.cluster_config,
                log_dir=log_dir,
                dependencies=dependencies,
                het_group=het_group,
                total_het_groups=total_het_groups,
                overlap=len(group.tasks) > 1,
                with_ray=self.global_config["with_ray"],
                extra_package_dirs=self.global_config["extra_package_dirs"],
            )
            executors.append(executor)
            het_group_indices.append(het_group)

            LOG.info(f"Task '{task.name}' command: {cmd}")

        # Create nemo-run scripts
        if len(commands) == 1:
            # Single task
            metadata = (
                {"use_with_ray_cluster": True}
                if self.global_config["with_ray"] and self.cluster_config["executor"] == "slurm"
                else None
            )
            return exp.add(
                run.Script(inline=commands[0], metadata=metadata),
                executor=executors[0],
                name="nemo-run",
                dependencies=None,  # Dependencies handled at executor level
            )
        else:
            # Multiple tasks (heterogeneous or parallel)
            if group.heterogeneous:
                executors[0].het_group_indices = het_group_indices

            metadata = (
                {"use_with_ray_cluster": True}
                if self.global_config["with_ray"] and self.cluster_config["executor"] == "slurm"
                else None
            )

            return exp.add(
                [
                    run.Script(inline=command, metadata=(metadata if idx == 0 else None))
                    for idx, command in enumerate(commands)
                ],
                executor=executors,
                name="nemo-run",
                dependencies=None,  # Dependencies handled at executor level
            )

    def _resolve_dependencies(self, group: TaskGroup) -> Optional[List[str]]:
        """Resolve task group dependencies to task handles."""
        if not group.dependencies:
            return None

        dependencies = []
        for dep in group.dependencies:
            if isinstance(dep, str):
                # External experiment dependency - would need to resolve handles
                # For now, simplified
                dependencies.append(dep)
            elif isinstance(dep, TaskGroup):
                # Internal dependency
                if dep._task_id:
                    dependencies.append(dep._task_id)

        return dependencies if dependencies else None
