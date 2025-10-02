import argparse
import logging
import os
from dataclasses import dataclass
from typing import Annotated

import httpx
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.clients import MCPStdioClient
from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    error: str | None = None
    result: str | None = None


mcp = FastMCP(name="brave")

# Populated from CLI args in main()
BRAVE_API_KEY: str | None = None


## See docs https://api-dashboard.search.brave.com/app/documentation/summarizer-search/get-started
@mcp.tool()
async def summarize(
    query: Annotated[str, Field(description="Search query.")],
) -> ExecutionResult:
    """Get a summary of search results from the web using Brave."""

    if not BRAVE_API_KEY:
        return {"error": "Missing Brave API key"}

    base_url = "https://api.search.brave.com/res/v1"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    async with httpx.AsyncClient() as client:
        ## First get search results.
        response = await client.get(f"{base_url}/web/search", params={"q": query, "summary": True}, headers=headers)
        if response.status_code != 200:
            return {"error": response.json()["error"]}

        ## Then get the summary.
        response = await client.get(
            f"{base_url}/summarizer/search", params={"key": response.json()["summarizer"]["key"]}, headers=headers
        )
        if response.status_code != 200:
            return {"error": response.json()["error"]}

    result = "".join([s["data"] for s in response.json()["summary"]])

    return {"result": result}


class BraveSearchTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.brave_tool"],
                },
                "hide_args": {},
                "init_hook": "nemo_skills.mcp.servers.brave_tool.stdio_connector",
            }
        )


def stdio_connector(client: MCPStdioClient):
    client.server_params = StdioServerParameters(
        command=client.server_params.command,
        args=list(client.server_params.args) + ["--api-key", os.getenv("BRAVE_API_KEY", "")],
    )


def main():
    parser = argparse.ArgumentParser(description="MCP server for Brave web search tool")
    parser.add_argument("--api-key", type=str, required=False, help="Brave API Key")
    args = parser.parse_args()

    global BRAVE_API_KEY
    # Prefer CLI arg; do not fall back to environment unless explicitly desired
    BRAVE_API_KEY = args.api_key

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
