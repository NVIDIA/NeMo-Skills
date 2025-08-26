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

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Annotated

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.utils import load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    process_status: str
    stderr: str
    stdout: str


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None


@mcp.tool()
async def execute(
    code: Annotated[str, Field(description="Code to run in python interpretter")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
) -> ExecutionResult:
    """Executes the given python code"""
    language = "ipython"
    try:
        output, _ = await sandbox.execute_code(code, language=language, timeout=timeout, session_id=session_id)
    except RemoteProtocolError:
        return {"process_status": "fail", "stdout": "", "stderr": f"Error connecting to sandbox"}
    return output


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    parser.add_argument("--config", dest="config", type=str, required=False, help="Path to OmegaConf YAML file")
    parser.add_argument(
        "--config-dir",
        dest="config_dir",
        type=str,
        required=False,
        help="Directory containing config file (Hydra-compatible)",
    )
    parser.add_argument(
        "--config-name",
        dest="config_name",
        type=str,
        default="config",
        help="Config file name without extension (Hydra-compatible)",
    )
    args = parser.parse_args()

    try:
        cfg = load_mcp_config(
            config=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
        )
    except ValueError as e:
        logger.warning(f"{e} Falling back to default local sandbox config.")
        cfg = OmegaConf.create({"sandbox": {"sandbox_type": "local"}})

    global sandbox
    sandbox_cfg = OmegaConf.to_container(cfg.sandbox, resolve=True)
    sandbox = get_sandbox(**sandbox_cfg)
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
