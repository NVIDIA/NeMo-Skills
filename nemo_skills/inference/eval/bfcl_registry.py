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

"""
BFCL Custom Tools Registry

This module provides a registry system for custom BFCL tools, allowing users to
register their own tool classes via JSON configuration.

Usage:
    Create ~/.bfcl_tools.json or set BFCL_TOOLS_CONFIG=/path/to/config.json

    Example config:
    {
        "tools": {
            "CalculatorAPI": "my_tools.calculator",
            "CustomAPI": "/path/to/custom_api.py"
        },
        "stateless_classes": ["MathUtils"]
    }

    Supports both module paths and file paths:
    - Module path: "my_module.submodule" (must be on Python path)
    - File path: "/absolute/path/to/file.py" (loaded dynamically)
"""

import os
import json
import logging
import importlib.util
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Global registry for custom tools
_CUSTOM_TOOL_REGISTRY: Dict[str, str] = {}
_CUSTOM_STATELESS_CLASSES: List[str] = []
_LOADED_MODULES: Dict[str, Any] = {}  # Cache for dynamically loaded modules
_REGISTRY_INITIALIZED = False


def _is_file_path(path: str) -> bool:
    """Check if the given path is a file path rather than a module path."""
    return os.path.isabs(path) or path.endswith('.py')


def _load_module_from_file(file_path: str, module_name: str) -> Any:
    """Dynamically load a Python module from a file path."""
    if file_path in _LOADED_MODULES:
        return _LOADED_MODULES[file_path]

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Python file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _LOADED_MODULES[file_path] = module
    LOG.info(f"Dynamically loaded module from file: {file_path}")
    return module


def get_tool_class(class_name: str, module_path: str) -> type:
    """
    Get a tool class, handling both module paths and file paths.

    Args:
        class_name: Name of the class to retrieve
        module_path: Either a Python module path or file path

    Returns:
        The requested class

    Raises:
        ImportError: If the module or class cannot be loaded
        AttributeError: If the class doesn't exist in the module
    """
    if _is_file_path(module_path):
        # Handle file path
        module = _load_module_from_file(module_path, f"_dynamic_{class_name.lower()}")
    else:
        # Handle module path
        module = importlib.import_module(module_path)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module at '{module_path}'")

    return getattr(module, class_name)


def _load_tools_from_config() -> None:
    """Load custom tools from JSON configuration file."""
    # Check for config file in order of preference
    config_paths = [
        os.getenv("BFCL_TOOLS_CONFIG"),
        os.path.expanduser("~/.bfcl_tools.json"),
        os.path.expanduser("~/.config/bfcl_tools.json"),
        "bfcl_tools.json"
    ]

    config_file = None
    for path in config_paths:
        if path and Path(path).exists():
            config_file = path
            break

    if not config_file:
        LOG.debug("No BFCL tools configuration file found")
        return

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        tools_dict = config.get("tools", {})
        stateless_classes = config.get("stateless_classes", [])

        global _CUSTOM_TOOL_REGISTRY, _CUSTOM_STATELESS_CLASSES
        _CUSTOM_TOOL_REGISTRY.update(tools_dict)
        _CUSTOM_STATELESS_CLASSES.extend(stateless_classes)

        LOG.info(f"Loaded {len(tools_dict)} tools from config file: {config_file}")

        # Validate file paths exist
        for class_name, module_path in tools_dict.items():
            if _is_file_path(module_path) and not Path(module_path).exists():
                LOG.warning(f"File path for {class_name} does not exist: {module_path}")

    except Exception as e:
        LOG.error(f"Error loading tools from config file {config_file}: {e}")
        raise


def _initialize_registry() -> None:
    """Initialize the registry by loading from JSON configuration."""
    global _REGISTRY_INITIALIZED

    if _REGISTRY_INITIALIZED:
        return

    _load_tools_from_config()
    _REGISTRY_INITIALIZED = True


def get_custom_tool_mapping() -> Dict[str, str]:
    """
    Get the current custom tool mapping.

    Returns:
        Dictionary mapping class names to module/file paths
    """
    _initialize_registry()
    return _CUSTOM_TOOL_REGISTRY.copy()


def get_custom_stateless_classes() -> List[str]:
    """
    Get the list of custom stateless classes.

    Returns:
        List of class names that are stateless
    """
    _initialize_registry()
    return _CUSTOM_STATELESS_CLASSES.copy()


def create_tools_config(output_file: str = "bfcl_tools.json") -> None:
    """
    Create a sample tools configuration file.

    Args:
        output_file: Path to output configuration file

    Example:
        create_tools_config("my_bfcl_tools.json")
    """
    sample_config = {
        "tools": {
            "CalculatorAPI": "custom_tools_example",
            "CustomFileSystem": "custom_tools_example",
            "MyAPI": "my_module.my_api",
            "WeatherAPI": "weather_tools.api",
            "CustomAPI": "/nemo_run/code/custom_api.py"
        },
        "stateless_classes": [
            "MathUtils",
            "StringHelpers"
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample BFCL tools configuration created: {output_file}")
    print(f"Edit this file and set BFCL_TOOLS_CONFIG={output_file} to use it")
    print("\nThe 'tools' section supports both:")
    print("  - Module paths: 'my_module.submodule' (must be on Python path)")
    print("  - File paths: '/absolute/path/to/file.py' (loaded dynamically)")


def list_registered_tools() -> None:
    """Print all currently registered tools."""
    _initialize_registry()

    print("🔧 BFCL Registered Custom Tools:")
    print("=" * 40)

    if not _CUSTOM_TOOL_REGISTRY:
        print("No custom tools registered.")
        print("\nTo register tools, create a JSON config file:")
        print("  ~/.bfcl_tools.json")
        print("  ~/.config/bfcl_tools.json")
        print("  Or set BFCL_TOOLS_CONFIG=/path/to/config.json")
        return

    for class_name, module_path in _CUSTOM_TOOL_REGISTRY.items():
        path_type = "file" if _is_file_path(module_path) else "module"
        stateless_marker = " (stateless)" if class_name in _CUSTOM_STATELESS_CLASSES else ""
        print(f"  {class_name} -> {module_path} [{path_type}]{stateless_marker}")

    print(f"\nTotal: {len(_CUSTOM_TOOL_REGISTRY)} custom tools registered")


def validate_tools_config(config_file: Optional[str] = None) -> bool:
    """
    Validate the tools configuration by attempting to load all registered classes.

    Args:
        config_file: Optional path to config file to validate

    Returns:
        True if all tools can be loaded successfully
    """
    if config_file:
        # Temporarily override the config file
        original_env = os.environ.get("BFCL_TOOLS_CONFIG")
        os.environ["BFCL_TOOLS_CONFIG"] = config_file

        # Reset registry to force reload
        global _REGISTRY_INITIALIZED, _CUSTOM_TOOL_REGISTRY, _CUSTOM_STATELESS_CLASSES
        _REGISTRY_INITIALIZED = False
        _CUSTOM_TOOL_REGISTRY.clear()
        _CUSTOM_STATELESS_CLASSES.clear()

    try:
        _initialize_registry()

        all_valid = True
        for class_name, module_path in _CUSTOM_TOOL_REGISTRY.items():
            try:
                get_tool_class(class_name, module_path)
                print(f"✅ {class_name}: Successfully loaded from {module_path}")
            except Exception as e:
                print(f"❌ {class_name}: Failed to load from {module_path} - {e}")
                all_valid = False

        return all_valid

    finally:
        if config_file:
            # Restore original environment
            if original_env:
                os.environ["BFCL_TOOLS_CONFIG"] = original_env
            else:
                os.environ.pop("BFCL_TOOLS_CONFIG", None)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        output_file = sys.argv[2] if len(sys.argv) > 2 else "bfcl_tools.json"
        create_tools_config(output_file)
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        list_registered_tools()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        config_file = sys.argv[2] if len(sys.argv) > 2 else None
        success = validate_tools_config(config_file)
        sys.exit(0 if success else 1)
    else:
        print("BFCL Tools Registry")
        print("Usage:")
        print("  python -m nemo_skills.inference.eval.bfcl_registry create-config [filename]")
        print("  python -m nemo_skills.inference.eval.bfcl_registry list")
        print("  python -m nemo_skills.inference.eval.bfcl_registry validate [config_file]")
