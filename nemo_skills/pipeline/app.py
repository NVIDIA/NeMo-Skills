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

import inspect
import logging
import os
from functools import wraps
from typing import Callable

import nemo_run as run
import typer
from typer.models import ParameterInfo

from nemo_skills.pipeline.utils import get_mounts_from_config, get_tunnel

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def wrap_arguments(arguments: str):
    """Returns a mock context object to allow using the cli entrypoints as functions."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    # first one is the cli name
    return MockContext(args=arguments.split(" "))


def typer_unpacker(f: Callable):
    """from https://github.com/fastapi/typer/issues/279"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get the default function argument that aren't passed in kwargs via the
        # inspect module: https://stackoverflow.com/a/12627202
        missing_default_values = {
            k: v.default
            for k, v in inspect.signature(f).parameters.items()
            if v.default is not inspect.Parameter.empty and k not in kwargs
        }

        for name, func_default in missing_default_values.items():
            # If the default value is a typer.Option or typer.Argument, we have to
            # pull either the .default attribute and pass it in the function
            # invocation, or call it first.
            if isinstance(func_default, ParameterInfo):
                if callable(func_default.default):
                    kwargs[name] = func_default.default()
                else:
                    kwargs[name] = func_default.default
                if kwargs[name] is ...:
                    raise TypeError(f"Missing required argument: '{name}'")

        # Call the wrapped function with the defaults injected if not specified.
        return f(*args, **kwargs)

    return wrapper


def create_remote_directory(directory: str | list, cluster_config: dict):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directory, str):
        directory = [directory]

    if cluster_config.get('executor') != 'slurm':
        tunnel = run.LocalTunnel(job_dir=directory[0])
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} in local filesystem.")
        tunnel.cleanup()

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = directory[0]

        tunnel = get_tunnel(cluster_config)
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} on remote cluster.")
        tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def check_remote_mount_directories(directories: list, cluster_config: dict, exit_on_failure: bool = True):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directories, str):
        directories = [directories]

    if cluster_config.get('executor') != 'slurm':
        tunnel = run.LocalTunnel(job_dir=None)

        all_dirs_exist = True
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        tunnel.cleanup()

        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = os.getcwd()

        tunnel = get_tunnel(cluster_config)
        missing_source_locations = []

        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        tunnel.cleanup()

        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def add_mount_path(mount_source: str, mount_dest: str, cluster_config):
    """Add a mount path to the cluster configuration."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if 'mounts' in cluster_config:
        original_mounts = get_mounts_from_config(cluster_config)
        added_mount = False
        for mount_path in original_mounts:
            source, destination = mount_path.split(':')

            if source == mount_source and destination == mount_dest:
                return

        if not added_mount:
            cluster_config['mounts'].append(f"{mount_source}:{mount_dest}")
            logging.info(f"Added mount path: `{mount_source}:{mount_dest}`")

    else:
        raise ValueError("No mounts found in cluster config, can only add to existing mount list.")
