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
Backward compatibility adapter for the old add_task function.
This allows existing code to continue working while gradually migrating to the new system.
"""

import logging
from typing import List, Optional, Union

import nemo_run as run

from nemo_skills.pipeline.utils.task_factories import TaskFactory
from nemo_skills.pipeline.utils.task_system import PipelineBuilder, TaskGroup
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def add_task_v2(
    exp: run.Experiment,
    cmd: Union[str, List[str]],
    task_name: str,
    cluster_config: dict,
    container: Union[str, List[str]],
    num_tasks: Union[int, List[int]] = 1,
    num_gpus: Optional[int] = None,
    num_nodes: int = 1,
    log_dir: Optional[str] = None,
    partition: Optional[str] = None,
    time_min: Optional[str] = None,
    with_sandbox: bool = False,
    sandbox_port: Optional[int] = None,
    server_config: Optional[dict] = None,
    reuse_code_exp: Union[str, run.Experiment, None] = None,
    reuse_code: bool = True,
    task_dependencies: Optional[List[str]] = None,
    run_after: Union[str, List[str], None] = None,
    get_server_command=None,
    extra_package_dirs: Optional[List[str]] = None,
    slurm_kwargs: Optional[dict] = None,
    heterogeneous: bool = False,
    with_ray: bool = False,
    installation_command: Optional[str] = None,
    skip_hf_home_check: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Backward compatibility adapter for the old add_task function.

    This function translates old add_task calls to use the new task system internally,
    while maintaining the same interface and return value.
    """

    # Create a temporary pipeline builder for this single task
    builder = PipelineBuilder(f"legacy_{task_name}", cluster_config)
    builder.set_global_config(
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        extra_package_dirs=extra_package_dirs,
        with_ray=with_ray,
    )

    # Create task group
    task_group = TaskGroup(task_name, heterogeneous=heterogeneous)

    # Handle server task if specified
    if server_config is not None and int(server_config.get("num_gpus", 0)) > 0:
        server_task = TaskFactory.create_server_task(
            name=f"{task_name}_server",
            server_type=server_config["server_type"],
            container=server_config.get("container", cluster_config["containers"][server_config["server_type"]]),
            num_gpus=server_config["num_gpus"],
            num_nodes=server_config.get("num_nodes", 1),
            server_args=server_config.get("server_args", ""),
            server_entrypoint=server_config.get("server_entrypoint"),
            model_path=server_config.get("model"),
            partition=partition,
        )
        task_group.add_task(server_task)

    # Handle main task(s)
    if cmd:
        if isinstance(cmd, str):
            cmd = [cmd]
        if isinstance(container, str):
            container = [container]
        if isinstance(num_tasks, int):
            num_tasks = [num_tasks]

        if len(cmd) != len(container) or len(cmd) != len(num_tasks):
            raise ValueError("Number of commands, containers and num_tasks must match.")

        for cur_idx, (cur_cmd, cur_container, cur_tasks) in enumerate(zip(cmd, container, num_tasks)):
            main_task = TaskFactory.create_generation_task(
                name=f"{task_name}_main" if len(cmd) == 1 else f"{task_name}_main_{cur_idx}",
                cmd=cur_cmd,
                container=cur_container,
                num_tasks=cur_tasks,
                num_gpus=num_gpus if server_config is None else 0,
                num_nodes=num_nodes,
                partition=partition,
                installation_command=installation_command,
            )

            # Apply slurm kwargs
            if slurm_kwargs:
                main_task.resources.slurm_kwargs.update(slurm_kwargs)

            task_group.add_task(main_task)

    # Handle sandbox task if specified
    if with_sandbox:
        sandbox_task = TaskFactory.create_sandbox_task(
            container=cluster_config["containers"]["sandbox"],
            port=sandbox_port,
        )
        task_group.add_task(sandbox_task)

    # Handle dependencies
    if run_after:
        if isinstance(run_after, (str, run.Experiment)):
            run_after = [run_after]
        for dep in run_after:
            task_group.add_dependency(dep)

    # Add to builder and build
    builder.add_task_group(task_group)

    # Build the experiment
    task_ids = builder.build_experiment(exp, log_dir or "/tmp")

    # Return the first task ID (matches old behavior)
    return task_ids[0] if task_ids else None


# Alias for backward compatibility
add_task_legacy = add_task_v2
