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
import os
from dataclasses import dataclass, field
from typing import Annotated

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    error: str | None = None
    results: str | None = None


mcp = FastMCP(name="exa_tool")


@mcp.tool()
async def exa_websearch(
    query: Annotated[str, Field(description="Search query for Exa.")],
) -> ExecutionResult:
    """Search the web using Exa. Provide relevant links in your answer."""

    url = "https://api.exa.ai/answer"
    headers = {
        "x-api-key": os.getenv("EXA_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {"query": f"{query}"}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": response.json()["error"]}
    else:
        return {"result": response.json()["answer"]}


def main(argv=None):
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
