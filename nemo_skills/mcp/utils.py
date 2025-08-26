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

import importlib
import importlib.util
import json
import logging
import os
import tempfile

from mcp import StdioServerParameters
from mcp.types import CallToolResult
from omegaconf import OmegaConf

from nemo_skills.mcp.clients import MCPStdioClient, MCPStreamableHttpClient

logger = logging.getLogger(__name__)


def locate(path):
    # If it's not a string, assume already an object and return
    if not isinstance(path, str):
        return path

    # Handle file path with ::attribute syntax
    if "::" in path:
        file_path, attr_name = path.split("::", 1)
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr_name)

    # Handle standard dotted module path
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def exa_auth_connector(client: MCPStreamableHttpClient):
    client.base_url = f"{client.base_url}?exaApiKey={os.getenv('EXA_API_KEY')}"


def exa_stdio_connector(client: MCPStdioClient):
    client.server_params = StdioServerParameters(
        command=client.server_params.command,
        args=list(client.server_params.args) + ["--exa-api-key", os.getenv("EXA_API_KEY", "")],
    )


def exa_output_formatter(result: CallToolResult):
    if getattr(result, "isError", False):
        logger.error(f"Exa error: {result}")
        return result.content[0].text
    return json.loads(result.content[0].text)["results"]


def hydra_config_connector_factory(config_obj):
    """Return a connector that writes the provided OmegaConf config to a temp YAML and injects Hydra args.

    Assumes `config_obj` is an OmegaConf DictConfig; we just OmegaConf.save and inject
    `--config-dir`/`--config-name`. Path and argument management only.
    """

    def connector(client: MCPStdioClient):
        temp_dir = tempfile.mkdtemp(prefix="mcp_cfg_")
        cfg_path = os.path.join(temp_dir, "config.yaml")
        OmegaConf.save(config_obj, cfg_path, resolve=True)

        client.server_params = StdioServerParameters(
            command=client.server_params.command,
            args=list(client.server_params.args) + ["--config-dir", temp_dir, "--config-name", "config"],
        )

    return connector
