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
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_manager import Tool
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

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
    code: Annotated[str, Field(description="Code to run in python interpreter")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
) -> ExecutionResult:
    """Executes the given python code"""
    language = "ipython"
    try:
        output, _ = await sandbox.execute_code(code, language=language, timeout=timeout, session_id=session_id)
    except RemoteProtocolError:
        return {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
    return output


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    add_config_args(parser)
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


# ==============================
# Module-based tool implementation
# ==============================


class PythonTool(Tool):
    provider_id = "python"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "exec_timeout_s": 10,
            # args to pass to MCP stdio server
            "command": "python",
            "args": ["-m", "nemo_skills.mcp.servers.python_tool"],
            # hide args from schemas and sanitize at runtime
            "hide_args": {"execute": ["session_id", "timeout"]},
        }
        # Lazy MCP client import to avoid circulars
        self._mcp_client = None

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        overrides = overrides or {}
        self._config.update(overrides)

        # Build MCP stdio client with hide_args applied
        from nemo_skills.mcp.clients import MCPStdioClient

        self._mcp_client = MCPStdioClient(
            command=self._config["command"],
            args=list(self._config["args"]),
            hide_args=self._config.get("hide_args"),
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        # Delegate to the MCP client to fetch authoritative schemas
        return await self._mcp_client.list_tools()

    async def execute(self, tool_name: str, arguments: Dict[str, Any]):
        if tool_name != "execute":
            raise ValueError(f"Unknown tool for PythonTool: {tool_name}")
        # Fallback timeout if not provided
        arguments = dict(arguments)
        arguments.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        return await self._mcp_client.call_tool("execute", arguments)

    async def shutdown(self) -> None:
        return None


if __name__ == "__main__":
    main()
