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

import logging
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from nemo_skills.code_execution.sandbox import LocalSandbox

logger = logging.getLogger(__name__)

mcp = FastMCP(name="python_tool")

sandbox = LocalSandbox()


@dataclass
class ExecutionResult:
    process_status: str
    stderr: str
    stdout: str

@mcp.tool()
async def execute(code: str) -> ExecutionResult:
    """Executes the given python code

    Args:
        code: the code to execute
    """
    language = "ipython"
    output, _ = await sandbox.execute_code(code, language=language)
    logger.info('Ran request with status: %s', output["process_status"])
    return output


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
