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

import json
import logging
from dataclasses import dataclass, field
from typing import Annotated

import hydra
from httpx import RemoteProtocolError
from hydra.core.config_store import ConfigStore
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    process_status: str
    stderr: str
    stdout: str


mcp = FastMCP(name="python_tool")

# Initialized from Hydra config in main()
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
    return json.loads(
        output["stdout"]
    )  # Sandbox bug fix: Remove extra layer of output wrapping, the result is a {"process_status": ..., "stdout": ..., "stderr": ...} dict


@dataclass
class PythonToolConfig:
    # Arbitrary sandbox config; e.g. {sandbox_type: local, host: 127.0.0.1, port: 6000}
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})


cs = ConfigStore.instance()
cs.store(name="base_python_tool_config", node=PythonToolConfig)


@hydra.main(version_base=None, config_name="base_python_tool_config")
def main(cfg: PythonToolConfig):
    global sandbox
    sandbox = get_sandbox(**cfg.sandbox)
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
