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
import os
from dataclasses import dataclass
from typing import Annotated

import httpx
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult
from pydantic import Field

from nemo_skills.mcp.clients import MCPStdioClient, MCPStreamableHttpClient
from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    error: str | None = None
    result: str | None = None


mcp = FastMCP(name="exa")

# Populated from CLI args in main()
EXA_API_KEY: str | None = None


@mcp.tool()
async def answer(
    query: Annotated[str, Field(description="Search query.")],
) -> ExecutionResult:
    """Search the web using Exa. Provide relevant links in your answer."""

    # Ensure API key is provided via CLI argument
    if not EXA_API_KEY:
        return {"error": "Missing Exa API key."}

    url = "https://api.exa.ai/answer"
    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"query": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": response.json()["error"]}

    return {"result": response.json()["answer"]}


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="MCP server for Exa web search tool")
    parser.add_argument("--api-key", type=str, required=False, help="Exa API key")
    args = parser.parse_args()

    global EXA_API_KEY
    # Prefer CLI arg; do not fall back to environment unless explicitly desired
    EXA_API_KEY = args.api_key

    # Initialize and run the server
    mcp.run(transport="stdio")


# ==============================
# Module-based tool implementation
# ==============================


def exa_stdio_connector(client: MCPStdioClient):
    client.server_params = StdioServerParameters(
        command=client.server_params.command,
        args=list(client.server_params.args) + ["--api-key", os.getenv("EXA_API_KEY", "")],
    )


class ExaTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        # Defaults for stdio Exa server launch using explicit client class
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.exa_tool"],
                },
                "hide_args": {},
                "init_hook": "nemo_skills.mcp.servers.exa_tool.exa_stdio_connector",
            }
        )


def exa_auth_connector(client: MCPStreamableHttpClient):
    client.base_url = f"{client.base_url}?exaApiKey={os.getenv('EXA_API_KEY')}"


def exa_output_formatter(result: CallToolResult):
    if getattr(result, "isError", False):
        logger.error(f"Exa error: {result}")
        return result.content[0].text
    return json.loads(result.content[0].text)["results"]


class ExaMCPTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        # Defaults for Exa hosted MCP over HTTP using explicit client class
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient",
                "client_params": {
                    "base_url": "https://mcp.exa.ai/mcp",
                },
                # Add API key via query param using the helper
                "init_hook": "nemo_skills.mcp.servers.exa_tool.exa_auth_connector",
                # Optional: limit to specific tools
                # "enabled_tools": ["web_search_exa"],
                # Parse structured output conveniently
                "output_formatter": "nemo_skills.mcp.servers.exa_tool.exa_output_formatter",
            }
        )


if __name__ == "__main__":
    main()
