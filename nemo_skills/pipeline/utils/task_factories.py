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

from typing import Dict, List, Optional

from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.pipeline.utils.task_system import (
    MainTask,
    ResourceSpec,
    SandboxTask,
    ServerTask,
    TaskGroup,
)


class TaskFactory:
    """Factory for creating common task types with sensible defaults."""

    @staticmethod
    def create_generation_task(
        name: str,
        cmd: str,
        container: str,
        num_gpus: int = 0,
        num_nodes: int = 1,
        num_tasks: int = 1,
        partition: Optional[str] = None,
        installation_command: Optional[str] = None,
        exclusive: bool = True,
        **kwargs,
    ) -> MainTask:
        """Create a generation/inference task."""
        resources = ResourceSpec(
            num_tasks=num_tasks,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
            exclusive=exclusive,
        )

        return MainTask(
            name=name,
            command=cmd,
            container=container,
            resources=resources,
            installation_command=installation_command,
            **kwargs,
        )

    @staticmethod
    def create_server_task(
        name: str,
        server_type: str,
        container: str,
        num_gpus: int,
        num_nodes: int = 1,
        model_path: Optional[str] = None,
        server_args: str = "",
        server_entrypoint: Optional[str] = None,
        server_port: Optional[int] = None,
        partition: Optional[str] = None,
        **kwargs,
    ) -> ServerTask:
        """Create a model server task."""
        resources = ResourceSpec(
            num_tasks=1,  # Will be updated by server command builder
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
        )

        return ServerTask(
            name=name,
            command="",  # Will be built by prepare_command
            container=container,
            resources=resources,
            server_type=server_type,
            server_args=server_args,
            server_entrypoint=server_entrypoint,
            model_path=model_path,
            server_port=server_port,
            log_prefix="server",
            **kwargs,
        )

    @staticmethod
    def create_sandbox_task(
        name: str = "sandbox", container: str = None, port: Optional[int] = None, **kwargs
    ) -> SandboxTask:
        """Create a code execution sandbox task."""
        if port is None:
            port = get_free_port(strategy="random")

        resources = ResourceSpec(
            num_tasks=1,
            num_gpus=0,
            num_nodes=1,
        )

        return SandboxTask(
            name=name,
            command="",  # Will be built by prepare_command
            container=container,
            resources=resources,
            port=port,
            log_prefix="sandbox",
            **kwargs,
        )

    @staticmethod
    def create_training_task(
        name: str,
        cmd: str,
        container: str,
        num_gpus: int,
        num_nodes: int = 1,
        num_tasks: int = 1,
        partition: Optional[str] = None,
        exclusive: bool = False,
        **kwargs,
    ) -> MainTask:
        """Create a training task."""
        resources = ResourceSpec(
            num_tasks=num_tasks,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
            exclusive=exclusive,
        )

        return MainTask(name=name, command=cmd, container=container, resources=resources, **kwargs)

    @staticmethod
    def create_evaluation_task(
        name: str,
        cmd: str,
        container: str,
        num_gpus: int = 1,
        num_nodes: int = 1,
        partition: Optional[str] = None,
        **kwargs,
    ) -> MainTask:
        """Create an evaluation task."""
        resources = ResourceSpec(
            num_tasks=1,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            partition=partition,
        )

        return MainTask(name=name, command=cmd, container=container, resources=resources, **kwargs)

    @staticmethod
    def create_preprocessing_task(
        name: str,
        cmd: str,
        container: str,
        num_nodes: int = 1,
        num_tasks: int = 1,
        exclusive: bool = True,  # CPU tasks usually need exclusive access
        partition: Optional[str] = None,
        **kwargs,
    ) -> MainTask:
        """Create a data preprocessing task (typically CPU-only)."""
        resources = ResourceSpec(
            num_tasks=num_tasks,
            num_gpus=0,  # Preprocessing is typically CPU-only
            num_nodes=num_nodes,
            partition=partition,
            exclusive=exclusive,
        )

        return MainTask(name=name, command=cmd, container=container, resources=resources, **kwargs)


class PipelineTemplates:
    """Pre-built pipeline templates for common use cases."""

    @staticmethod
    def create_generation_with_server_pipeline(
        name: str,
        generation_cmd: str,
        server_type: str,
        model_path: str,
        cluster_config: Dict,
        server_gpus: int = 8,
        server_nodes: int = 1,
        server_port: Optional[int] = None,
        server_args: str = "",
        server_entrypoint: Optional[str] = None,
        with_sandbox: bool = False,
        sandbox_port: Optional[int] = None,
    ) -> TaskGroup:
        """Create a pipeline with server + generation task + optional sandbox."""

        group = TaskGroup(name, heterogeneous=True)

        # Add server task with specified port
        server_task = TaskFactory.create_server_task(
            name=f"{name}_server",
            server_type=server_type,
            container=cluster_config["containers"].get(server_type, cluster_config["containers"]["nemo"]),
            num_gpus=server_gpus,
            num_nodes=server_nodes,
            model_path=model_path,
            server_port=server_port,
            server_args=server_args,
            server_entrypoint=server_entrypoint,
        )
        group.add_task(server_task)

        # Add main generation task
        generation_task = TaskFactory.create_generation_task(
            name=f"{name}_generation",
            cmd=generation_cmd,
            container=cluster_config["containers"]["nemo-skills"],
            num_gpus=0,  # Generation uses server, no local GPUs needed
        )
        group.add_task(generation_task)

        # Add sandbox if requested
        if with_sandbox:
            sandbox_task = TaskFactory.create_sandbox_task(
                container=cluster_config["containers"]["sandbox"],
                port=sandbox_port,
            )
            group.add_task(sandbox_task)

        return group

    @staticmethod
    def create_multi_stage_training_pipeline(
        preprocessing_cmd: str,
        training_cmd: str,
        evaluation_cmd: str,
        cluster_config: Dict,
        training_gpus: int = 8,
        training_nodes: int = 1,
    ) -> List[TaskGroup]:
        """Create a multi-stage training pipeline: preprocess -> train -> evaluate."""

        # Stage 1: Preprocessing
        prep_group = TaskGroup("preprocessing")
        prep_task = TaskFactory.create_preprocessing_task(
            name="data_preprocessing",
            cmd=preprocessing_cmd,
            container=cluster_config["containers"]["nemo-skills"],
            num_nodes=2,  # Use multiple nodes for large datasets
        )
        prep_group.add_task(prep_task)

        # Stage 2: Training
        train_group = TaskGroup("training")
        train_task = TaskFactory.create_training_task(
            name="model_training",
            cmd=training_cmd,
            container=cluster_config["containers"]["nemo"],
            num_gpus=training_gpus,
            num_nodes=training_nodes,
        )
        train_group.add_task(train_task)
        train_group.add_dependency(prep_group)

        # Stage 3: Evaluation
        eval_group = TaskGroup("evaluation")
        eval_task = TaskFactory.create_evaluation_task(
            name="model_evaluation",
            cmd=evaluation_cmd,
            container=cluster_config["containers"]["nemo-skills"],
            num_gpus=1,
        )
        eval_group.add_task(eval_task)
        eval_group.add_dependency(train_group)

        return [prep_group, train_group, eval_group]
