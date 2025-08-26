import copy
from typing import Any

from omegaconf import DictConfig

from nemo_skills.mcp.clients import MCPClientManager
from nemo_skills.mcp.utils import locate

RESOLVABLE_PARAM_KEYS = {"output_formatter", "init_hook"}


def _resolve_special(value: Any, full_cfg: DictConfig) -> Any:
    if isinstance(value, str) and value == "@@full_config":
        return full_cfg
    return value


def _resolve_locate_mapping(spec: dict, full_cfg: DictConfig) -> Any:
    target = locate(spec.get("$locate"))
    raw_args = spec.get("args", [])
    raw_kwargs = spec.get("kwargs", {})

    # Recursively resolve nested $locate and @@full_config in args/kwargs
    args = [resolve_value(a, full_cfg) for a in raw_args]
    kwargs = {k: resolve_value(v, full_cfg) for k, v in raw_kwargs.items()}

    return target(*args, **kwargs)


def resolve_value(value: Any, full_cfg: DictConfig) -> Any:
    if isinstance(value, dict) and "$locate" in value:
        return _resolve_locate_mapping(value, full_cfg)
    return _resolve_special(value, full_cfg)


def resolve_adapters(cfg: DictConfig):
    adapters_cfg = cfg.adapters
    schema_adapter_obj = locate(adapters_cfg.schema_adapter)
    call_interpreter_obj = locate(adapters_cfg.call_interpreter)
    response_formatter_obj = locate(adapters_cfg.response_formatter)

    schema_adapter = schema_adapter_obj() if isinstance(schema_adapter_obj, type) else schema_adapter_obj
    call_interpreter = call_interpreter_obj() if isinstance(call_interpreter_obj, type) else call_interpreter_obj
    response_formatter = (
        response_formatter_obj() if isinstance(response_formatter_obj, type) else response_formatter_obj
    )
    return schema_adapter, call_interpreter, response_formatter


def build_client_manager(cfg: DictConfig) -> MCPClientManager:
    manager = MCPClientManager()
    for t in cfg.tools:
        client_cls = locate(t.client)
        # Copy and resolve parameters
        params = {} if t.get("params") is None else copy.deepcopy(dict(t.params))
        resolved_params = {}
        for key, val in params.items():
            if isinstance(val, dict) and "$locate" in val:
                resolved_params[key] = _resolve_locate_mapping(val, cfg)
            elif key in RESOLVABLE_PARAM_KEYS and isinstance(val, str):
                resolved_params[key] = locate(val)
            else:
                resolved_params[key] = resolve_value(val, cfg)

        client = client_cls(**resolved_params)
        manager.register(t.id, client)
    return manager
