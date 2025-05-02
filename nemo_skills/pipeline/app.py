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

from nemo_skills.pipeline.utils import (
    get_mounts_from_config,
    get_tunnel,
    add_mount_path,
    create_remote_directory,
    check_remote_mount_directories,
)  # noqa # pylint: disable=unused-import

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


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
