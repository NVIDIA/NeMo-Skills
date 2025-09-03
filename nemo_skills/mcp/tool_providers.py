from __future__ import annotations

from typing import Any, Dict, List

from nemo_skills.mcp.tool_manager import Tool
from nemo_skills.mcp.utils import locate


class MCPClientTool(Tool):
    """Base Tool that delegates to an MCP client (stdio or streamable HTTP).

    Config keys (overridable via tool_overrides[provider_id]):
      - transport: "stdio" | "streamable_http" (default: stdio)
      - command, args: for stdio transport
      - base_url: for streamable_http transport
      - hide_args, disabled_tools, enabled_tools
      - output_formatter: dotted path or callable
      - init_hook: dotted path or callable (receives client for side effects)
    """

    provider_id: str = ""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "transport": "stdio",
            "command": "python",
            "args": [],
            "base_url": None,
            "hide_args": {},
            "disabled_tools": [],
            "enabled_tools": [],
            "output_formatter": None,
            "init_hook": None,
        }
        self._client = None

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def _resolve_maybe_callable(self, value: Any):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return locate(value)
            except Exception:
                # If not resolvable, pass through; client wrapper may handle
                return value
        return value

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        from nemo_skills.mcp.clients import MCPStdioClient, MCPStreamableHttpClient

        cfg = dict(self._config)
        if overrides:
            cfg.update(overrides)

        output_formatter = self._resolve_maybe_callable(cfg.get("output_formatter"))
        init_hook = self._resolve_maybe_callable(cfg.get("init_hook"))

        if cfg.get("transport", "stdio") == "streamable_http":
            self._client = MCPStreamableHttpClient(
                base_url=cfg["base_url"],
            )
            # Inject common behaviors set by metaclass wrapper
            # Note: MCPStreamableHttpClient __init__ doesn't accept these; we attach after
            self._client._hide_args = cfg.get("hide_args", {})
            self._client._disabled_tools = set(cfg.get("disabled_tools", []))
            self._client._enabled_tools = set(cfg.get("enabled_tools", []))
            self._client.output_formatter = output_formatter
            if callable(init_hook):
                init_hook(self._client)
        else:
            self._client = MCPStdioClient(
                command=cfg["command"],
                args=list(cfg.get("args", [])),
                hide_args=cfg.get("hide_args"),
                disabled_tools=cfg.get("disabled_tools"),
                enabled_tools=cfg.get("enabled_tools"),
                output_formatter=output_formatter,
                init_hook=init_hook,
            )

        self._config = cfg

    async def list_tools(self) -> List[Dict[str, Any]]:
        return await self._client.list_tools()

    async def execute(self, tool_name: str, arguments: Dict[str, Any]):
        return await self._client.call_tool(tool_name, arguments)
