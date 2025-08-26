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
import copy
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


def _process_hide_args(result, hide_args):
    if hide_args:
        output = []
        for entry in result:
            method_name = entry.get("name")
            schema = copy.deepcopy(entry.get("input_schema", {}))
            if method_name in hide_args and "properties" in schema:
                for arg in hide_args[method_name]:
                    schema["properties"].pop(arg, None)
                    if "required" in schema and arg in schema["required"]:
                        schema["required"].remove(arg)
            new_entry = dict(entry)
            new_entry["input_schema"] = schema
            output.append(new_entry)
        return output
    return result


def _filter_tools(result, disabled_tools, enabled_tools):
    if not isinstance(result, list):
        return result
    disabled_set = set(disabled_tools or [])
    enabled_set = set(enabled_tools or [])

    filtered = []
    names_seen = set()
    for entry in result:
        name = entry.get("name")
        if name is None:
            continue
        names_seen.add(name)
        if name in disabled_set:
            continue
        if enabled_set and name not in enabled_set:
            continue
        filtered.append(entry)

    if enabled_set:
        missing = enabled_set - names_seen
        if missing:
            raise ValueError(f"Enabled tools not found: {sorted(missing)}")

    return filtered


def async_wrapper(method):
    async def wrapped(self, *args, **kwargs):
        hide_args = kwargs.pop("hide_args", None)
        if hide_args is None:
            hide_args = getattr(self, "_hide_args", {})
        disabled_tools = kwargs.pop("disabled_tools", None)
        if disabled_tools is None:
            disabled_tools = getattr(self, "_disabled_tools", set())
        enabled_tools = kwargs.pop("enabled_tools", None)
        if enabled_tools is None:
            enabled_tools = getattr(self, "_enabled_tools", set())
        result = await method(self, *args, **kwargs)
        result = _process_hide_args(result, hide_args)
        result = _filter_tools(result, disabled_tools, enabled_tools)
        return result

    return wrapped


def _sanitize_input_args_for_tool(args_dict, tool_name, hide_args):
    """Remove hidden-argument keys from the model-supplied tool args.

    Only top-level keys are removed, matching how schemas are exposed. Returns
    a new dict when sanitization is needed; otherwise returns the original.
    """
    if not tool_name or not hide_args or not isinstance(args_dict, dict):
        return args_dict
    hidden_keys = set(hide_args.get(tool_name, []) or [])
    if not hidden_keys:
        return args_dict
    return {k: v for k, v in args_dict.items() if k not in hidden_keys}


def _wrap_call_tool_output_formatter(method):
    async def wrapped(self, *args, **kwargs):
        result = await method(self, *args, **kwargs)
        output_formatter = getattr(self, "output_formatter", None)
        if callable(output_formatter):
            return output_formatter(result)
        return result

    return wrapped


def inject_hide_args(init_func):
    @functools.wraps(init_func)
    def wrapper(
        self,
        *args,
        hide_args=None,
        disabled_tools=None,
        enabled_tools=None,
        output_formatter: Callable | None = None,
        init_hook: Callable | None = None,
        **kwargs,
    ):
        self._hide_args = hide_args or {}
        # Store as sets for fast membership checks
        self._disabled_tools = set(disabled_tools or [])
        self._enabled_tools = set(enabled_tools or [])
        # Optional common behaviors
        self.output_formatter = output_formatter
        self._init_hook = init_hook
        instance = init_func(self, *args, **kwargs)
        # Run init hook if provided (sync callable)
        if callable(self._init_hook):
            try:
                self._init_hook(self)
            except Exception:
                # Propagate to surface hook errors
                raise
        return instance

    return wrapper


class MCPClientMeta(type):
    """Metaclass that adds `hide_args` support to MCP client implementations.

    Responsibilities:
    - Wraps `__init__` to accept an optional `hide_args` mapping and stores it
      on instances as `_hide_args`.
    - Wraps `list_tools` so its returned tool schemas are post-processed to
      remove arguments specified in `_hide_args` (by pruning properties and
      updating `required`).
    - Input sanitization is available via `sanitize` method on the client.
    - Ensures every instance has a default `_hide_args` attribute even when
      subclasses do not define/override `__init__`.

    Example:
    ```python
    from typing import Any, Dict

    # Inherit from any MCPClient class (its class was created with MCPClientMeta)
    class MyClient(MCPClient):
        # No need to add `hide_args` to the signature or handle masking yourself
        ...

    # Consumers can now pass `hide_args` seamlessly. The metaclass-injected
    # logic ensures hidden parameters are pruned from tool input schemas
    # returned by list_tools. You can also disable specific tools or allow only
    # a subset using `disabled_tools` or `enabled_tools`. Additionally, all
    # MCP clients accept `output_formatter` and `init_hook` kwargs:
    # - output_formatter: Callable that post-processes tool call results
    # - init_hook: Callable that is invoked after instance initialization
    client = MyClient(
        base_url="https://mcp.example.com",
        hide_args={"tool_name": ["timeout"]},
        disabled_tools=["disallowed_tool"],
        enabled_tools=["tool_name", "safe_other_tool"],
        output_formatter=lambda r: (r.structuredContent or r.content),
        init_hook=lambda self: setattr(self, "_ready", True),
    )
    tools: list[Dict[str, Any]] = await client.list_tools()
    # The input_schema for "tool_name" will no longer expose
    # the "timeout" parameter.
    #
    # The model could still hypothetically call the tool with the hidden argument,
    # but as long as the sanitize method is called, the hidden argument will be
    # removed from the input schema.
    safe_args = client.sanitize("tool_name", {"timeout": 100000, "x": 1})
    result = await client.call_tool("tool_name", timeout=10, **safe_args)

    ```
    """

    def __new__(mcls, name, bases, namespace):
        orig_init = namespace.get("__init__")
        if orig_init is not None:
            namespace["__init__"] = inject_hide_args(orig_init)
        # Wrap list_tools for hide_args masking (async or sync)
        orig_list = namespace.get("list_tools")
        if orig_list is not None:
            wrapper = async_wrapper(orig_list)
            namespace["list_tools"] = wrapper

        # Wrap call_tool to apply output_formatter automatically
        orig_call_tool = namespace.get("call_tool")
        if orig_call_tool is not None:
            namespace["call_tool"] = _wrap_call_tool_output_formatter(orig_call_tool)

        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        # Create the instance using normal init flow
        instance = super().__call__(*args, **kwargs)
        # Add default attributes if they do not exist yet
        if not hasattr(instance, "_hide_args"):
            instance._hide_args = {}
        if not hasattr(instance, "_disabled_tools"):
            instance._disabled_tools = set()
        if not hasattr(instance, "_enabled_tools"):
            instance._enabled_tools = set()
        return instance


class MCPClient(metaclass=MCPClientMeta):
    """Abstract base for Model Context Protocol (MCP) clients.

    This base class defines the minimal interface used by the tool runtime:
    - list_tools(): Return a list of tool descriptors with `name`, `description`, and
      `input_schema` (JSON Schema) fields.
    - call_tool(tool, args): Invoke a tool by name with a dict of arguments.

    Common configurables are injected by the metaclass and can be passed to any
    concrete MCP client constructor without changing its signature:
    - hide_args: Dict[str, list[str]] mapping tool name -> argument keys to hide.
      Hidden keys are pruned from input schemas returned by list_tools().
    - disabled_tools: Iterable[str] of tool names to disable.
    - enabled_tools: Iterable[str] of tool names to allow (acts as an allowlist).
    - output_formatter: Optional callable that post-processes results of call_tool().
    - init_hook: Optional callable run after client initialization (e.g., to inject
      credentials or wire connectors).

    Example (manual usage):
    ```python
    client = MCPStdioClient(
        command="python",
        args=["-m", "nemo_skills.mcp.servers.python_tool"],
        hide_args={"execute": ["session_id", "timeout"]},
    )
    tools = await client.list_tools()
    safe = client.sanitize("execute", {"code": "print(1)", "timeout": 999})
    result = await client.call_tool("execute", safe)
    ```

    Example (configured via OmegaConf, similar to mcp_demo.py):
    ```python
    from omegaconf import OmegaConf
    from nemo_skills.mcp.config_loader import build_client_manager

    cfg = OmegaConf.create({
        "tools": [
            {
                "id": "python",
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.python_tool"],
                    "hide_args": {"execute": ["session_id", "timeout"]},
                },
            },
        ]
    })
    manager = build_client_manager(cfg)
    tools = await manager.list_all_tools()
    ```
    """

    # Manual sanitization helpers (input-only)
    def sanitize(self, tool: str, args: dict) -> dict:
        """Return a copy of args with hidden keys removed for the given tool."""
        return _sanitize_input_args_for_tool(args, tool, self._hide_args)

    @abstractmethod
    async def list_tools(self):
        pass

    @abstractmethod
    async def call_tool(self, tool: str, args: dict) -> Any:
        pass

    # Enforcement helpers
    def _assert_tool_allowed(self, tool: str):
        if tool in getattr(self, "_disabled_tools", set()):
            raise PermissionError(f"Tool '{tool}' is disabled")
        enabled = getattr(self, "_enabled_tools", set())
        if enabled and tool not in enabled:
            raise PermissionError(f"Tool '{tool}' is not in enabled_tools: {sorted(enabled)}")


class MCPStreamableHttpClient(MCPClient):
    """MCP client that connects to servers over the Streamable HTTP transport.

    Args:
        base_url: Base URL of the MCP server, e.g. "https://host:port/mcp".

    Behavior:
    - list_tools() fetches tool metadata from the server and normalizes schema
      field names (supports both input_schema and inputSchema).
    - call_tool() returns the server's structuredContent when present, otherwise
      returns the raw result object.

    The following optional configurables can be supplied (injected by the
    metaclass): hide_args, disabled_tools, enabled_tools, output_formatter,
    init_hook.

    Example (OmegaConf config, like in mcp_demo.py):
    ```python
    from omegaconf import OmegaConf
    from nemo_skills.mcp.config_loader import build_client_manager

    cfg = OmegaConf.create({
        "tools": [
            {
                "id": "exa_mcp",
                "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient",
                "params": {
                    "base_url": "https://mcp.exa.ai/mcp",
                    "enabled_tools": ["web_search_exa"],
                    "output_formatter": "nemo_skills.mcp.utils.exa_output_formatter",
                    "init_hook": "nemo_skills.mcp.utils.exa_auth_connector",
                },
            }
        ]
    })
    manager = build_client_manager(cfg)
    tools = await manager.list_all_tools()
    result = await manager.execute_tool("exa_mcp.web_search_exa", {"query": "nemo skills"})
    ```

    Example (manual usage):
    ```python
    client = MCPStreamableHttpClient(base_url="https://mcp.example.com/mcp")
    tools = await client.list_tools()
    result = await client.call_tool("some_tool", {"x": 1})
    ```
    """

    def __init__(self, base_url: str):
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
        self._assert_tool_allowed(tool)
        async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=args)
                return struct if (struct := result.structuredContent) is not None else result


class MCPStdioClient(MCPClient):
    """MCP client that launches a server via stdio.

    Args:
        command: Executable to launch (e.g., "python").
        args: Command-line arguments (e.g., ["-m", "nemo_skills.mcp.servers.python_tool"]).

    Behavior:
    - list_tools() fetches tool metadata from the running stdio server.
    - call_tool() returns the server's structuredContent.

    The following optional configurables can be supplied (injected by the
    metaclass): hide_args, disabled_tools, enabled_tools, output_formatter,
    init_hook.

    Example (OmegaConf config, like in mcp_demo.py):
    ```python
    from omegaconf import OmegaConf
    from nemo_skills.mcp.config_loader import build_client_manager

    cfg = OmegaConf.create({
        "tools": [
            {
                "id": "python",
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.python_tool"],
                    "hide_args": {"execute": ["session_id", "timeout"]},
                    "init_hook": {
                        "$locate": "nemo_skills.mcp.utils.hydra_config_connector_factory",
                        "kwargs": {"config_obj": "@@full_config"},
                    },
                },
            },
            {
                "id": "exa",
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.exa_tool"],
                    "init_hook": "nemo_skills.mcp.utils.exa_stdio_connector",
                },
            },
        ]
    })
    manager = build_client_manager(cfg)
    tools = await manager.list_all_tools()
    # Example tool call
    result = await manager.execute_tool("python.execute", {"code": "print(1)"})
    ```

    Example (manual usage):
    ```python
    client = MCPStdioClient(command="python", args=["-m", "nemo_skills.mcp.servers.python_tool"])
    tools = await client.list_tools()
    result = await client.call_tool("execute", {"code": "print(1)"})
    ```
    """

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
        self._assert_tool_allowed(tool)
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=args)
                return result.structuredContent


class MCPClientManager:
    """Registry and orchestrator for multiple MCP clients.

    Responsibilities:
    - register(name, client): Add a client under a unique name.
    - list_all_tools(): Merge tool listings across clients; names are qualified
      as "{client}.{tool}" to avoid collisions.
    - execute_tool(tool_name, args): Route a qualified tool call to the owning
      client (stripping the client prefix automatically).

    Example (using config loader, like in mcp_demo.py):
    ```python
    from omegaconf import OmegaConf
    from nemo_skills.mcp.config_loader import build_client_manager

    cfg = OmegaConf.create({
        "tools": [
            {"id": "python", "client": "nemo_skills.mcp.clients.MCPStdioClient", "params": {"command": "python", "args": ["-m", "nemo_skills.mcp.servers.python_tool"]}},
            {"id": "exa_mcp", "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient", "params": {"base_url": "https://mcp.exa.ai/mcp"}},
        ]
    })
    manager = build_client_manager(cfg)
    tools = await manager.list_all_tools()
    out = await manager.execute_tool("python.execute", {"code": "print('ok')"})
    ```

    Example (manual registration):
    ```python
    manager = MCPClientManager()
    manager.register("python", MCPStdioClient(command="python", args=["-m", "nemo_skills.mcp.servers.python_tool"]))
    tools = await manager.list_all_tools()
    out = await manager.execute_tool("python.execute", {"code": "print('ok')"})
    ```
    """

    def __init__(self):
        self.clients = {}
        self.tool_map: dict[str, str] = {}  # maps "client.tool" -> client_name
        self._tools_cache: list[dict[str, Any]] | None = None

    def register(self, name: str, client: MCPClient):
        # Enforce uniqueness of client (top-level) ids
        if name in self.clients:
            raise ValueError(f"Client name already registered: {name}")
        self.clients[name] = client
        for tool in client.tools:
            raw_tool_name = tool.get("name")
            if raw_tool_name is None:
                continue
            full_tool_name = f"{name}.{raw_tool_name}"
            self.tool_map[full_tool_name] = name

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
                raw_name = t["name"]
                full_name = f"{client_name}.{raw_name}"
                # Use full name in the merged listing
                if full_name in all_tools:
                    raise ValueError(f"Duplicate fully-qualified tool id detected: '{full_name}'")
                merged_tool = {"server": client_name, **t, "name": full_name}
                all_tools[full_name] = merged_tool

        self._tools_cache = list(all_tools.values())

        # Update tool_map for fast resolution
        self.tool_map = {t["name"]: t["server"] for t in self._tools_cache}

        return self._tools_cache

    def get_client_for_tool(self, tool_name: str) -> MCPClient:
        if "." not in tool_name:
            raise ValueError(f"Tool name must be in 'client.tool' format. Received: '{tool_name}'")
        client_name, _ = tool_name.split(".", 1)
        client = self.clients.get(client_name)
        if client is None:
            raise ValueError(f"No client registered with name {client_name}")
        return client

    async def execute_tool(self, tool_name: str, args: dict):
        client = self.get_client_for_tool(tool_name)
        # Strip client prefix before delegating to the underlying client
        bare_tool_name = tool_name.split(".", 1)[1] if "." in tool_name else tool_name
        return await client.call_tool(bare_tool_name, args)
