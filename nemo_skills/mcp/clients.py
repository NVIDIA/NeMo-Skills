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
import aiohttp
from typing import Dict, Any, List

class MCPHttpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.tools: List[Dict[str, Any]] = []

    async def list_tools(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/tools") as resp:
                self.tools = await resp.json()
                return self.tools

    async def call_tool(self, tool: str, args: dict) -> Any:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/call", json={"tool": tool, "args": args}) as resp:
                return await resp.json()


class MCPClientManager:
    def __init__(self):
        self.clients = {}
        self.tool_map: dict[str, str] = {}  # maps tool_name -> client_name (latest registered wins)
        self._tools_cache: list[dict[str, Any]] | None = None

    def register(self, name: str, client: MCPHttpClient):
        self.clients[name] = client
        for tool in client.tools:
            tool_name = tool["name"]
            self.tool_map[tool_name] = name  # latest registered overrides

    def get_client(self, name: str):
        return self.clients.get(name)

    async def list_all_tools(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """
        Return merged tool list from all clients.
        Most recently registered clients override earlier ones for tool name collisions.
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        all_tools: dict[str, dict[str, Any]] = {}
        for client_name, client in self.clients.items():
            tools = await client.list_tools()
            for t in tools:
                # Latest registered client wins for each tool
                all_tools[t["name"]] = {"server": client_name, **t}

        self._tools_cache = list(all_tools.values())

        # Update tool_map for fast resolution
        self.tool_map = {t["name"]: t["server"] for t in self._tools_cache}

        return self._tools_cache

    def get_client_for_tool(self, tool_name: str) -> MCPHttpClient:
        if tool_name not in self.tool_map:
            raise ValueError(f"No client registered for tool {tool_name}")
        client_name = self.tool_map[tool_name]
        return self.clients[client_name]

    async def execute_tool(self, tool_name: str, args: dict):
        client = self.get_client_for_tool(tool_name)
        return await client.call_tool(tool_name, args)

