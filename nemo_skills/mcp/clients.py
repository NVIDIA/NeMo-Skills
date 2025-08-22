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
from abc import abstractmethod, ABC
import aiohttp
import functools
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

import copy

def _process_hide_args(result, hide_args):
    if hide_args:
        output = []
        for entry in result:
            method_name = entry.get('name')
            schema = copy.deepcopy(entry.get('input_schema', {}))
            if method_name in hide_args and 'properties' in schema:
                for arg in hide_args[method_name]:
                    schema['properties'].pop(arg, None)
                    if 'required' in schema and arg in schema['required']:
                        schema['required'].remove(arg)
            new_entry = dict(entry)
            new_entry['input_schema'] = schema
            output.append(new_entry)
        return output
    return result


def async_wrapper(method):
    async def wrapped(self, *args, **kwargs):
        hide_args = kwargs.pop('hide_args', None)
        if hide_args is None:
            hide_args = getattr(self, '_hide_args', {})
        result = await method(self, *args, **kwargs)
        return _process_hide_args(result, hide_args)
    return wrapped

def inject_hide_args(init_func):
    @functools.wraps(init_func)
    def wrapper(self, *args, hide_args=None, **kwargs):
        self._hide_args = hide_args or {}
        return init_func(self, *args, **kwargs)
    return wrapper

class MCPClientMeta(type):
    def __new__(mcls, name, bases, namespace):
        # orig = namespace.get("list_tools")
        # if orig is not None:
        #     namespace["list_tools"] = async_wrapper(orig)
        # return super().__new__(mcls, name, bases, namespace)
                # Wrap __init__ for _hide_args injection
        orig_init = namespace.get('__init__')
        if orig_init is not None:
            namespace['__init__'] = inject_hide_args(orig_init)
        # Wrap list_tools for hide_args masking (async or sync)
        orig_list = namespace.get("list_tools")
        if orig_list is not None:
            wrapper = async_wrapper(orig_list)
            namespace["list_tools"] = wrapper

        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        # Create the instance using normal init flow
        instance = super().__call__(*args, **kwargs)
        # Add default attributes if they do not exist yet
        if not hasattr(instance, '_hide_args'):
            instance._hide_args = {}
        return instance

class MCPClient(metaclass=MCPClientMeta):
    def __init__(self, hide_args=None, **kwargs):
        self._hide_args = hide_args or {}
        super().__init__(**kwargs)

    @abstractmethod
    async def list_tools(self):
        pass

    @abstractmethod
    async def call_tool(self, tool: str, args: dict) -> Any:
        pass


class MCPHttpClient(MCPClient):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.tools: List[Dict[str, Any]] = []

    async def list_tools(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/list_tools") as resp:
                self.tools = await resp.json()
                return self.tools

    async def call_tool(self, tool: str, args: dict) -> Any:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/call_tool", json={"tool": tool, "args": args}) as resp:
                return await resp.json()


class MCPStreamableHttpClient(MCPClient):
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url
        self.tools: List[Dict[str, Any]] = []

    async def list_tools(self):
        async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools_list: List[Dict[str, Any]] = []
                # tools_resp.tools is expected to be a list of Tool objects
                for t in getattr(tools_resp, "tools", []) or []:
                    # Support both input_schema (python) and inputSchema (wire)
                    input_schema = getattr(t, "input_schema", None)
                    if input_schema is None:
                        input_schema = getattr(t, "inputSchema", None)
                    tools_list.append(
                        {
                            "name": getattr(t, "name", None),
                            "description": getattr(t, "description", ""),
                            "input_schema": input_schema,
                        }
                    )
                self.tools = tools_list
                return self.tools

    async def call_tool(self, tool: str, args: dict) -> Any:
        async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=args)
                return result.structuredContent

class MCPStdioClient(MCPClient):
    def __init__(self, command: str, args: list[str] | None = None):
        if args is None:
            args = []
        self.server_params = StdioServerParameters(command=command, args=args)
        self.tools: List[Dict[str, Any]] = []

    async def list_tools(self):
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools_list: List[Dict[str, Any]] = []
                for t in getattr(tools_resp, "tools", []) or []:
                    input_schema = getattr(t, "input_schema", None)
                    if input_schema is None:
                        input_schema = getattr(t, "inputSchema", None)
                    tools_list.append(
                        {
                            "name": getattr(t, "name", None),
                            "description": getattr(t, "description", ""),
                            "input_schema": input_schema,
                        }
                    )
                self.tools = tools_list
                return self.tools

    async def call_tool(self, tool: str, args: dict) -> Any:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=args)
                return result.structuredContent

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

