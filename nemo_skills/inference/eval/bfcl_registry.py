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
register their own tool classes without modifying the core BFCL code.

Usage:
    # Method 1: Programmatic registration
    from nemo_skills.inference.eval.bfcl_registry import register_tool_class
    register_tool_class("MyAPI", "my_module")

    # Method 2: Environment variable
    export BFCL_CUSTOM_TOOLS="MyAPI:my_module,AnotherAPI:another_module"

    # Method 3: Config file
    Create ~/.bfcl_tools.json or set BFCL_TOOLS_CONFIG=/path/to/config.json
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Global registry for custom tools
_CUSTOM_TOOL_REGISTRY: Dict[str, str] = {}
_CUSTOM_STATELESS_CLASSES: List[str] = []
_REGISTRY_INITIALIZED = False


def register_tool_class(class_name: str, module_path: str, stateless: bool = False) -> None:
    """
    Register a custom tool class with the BFCL system.

    Args:
        class_name: Name of the tool class (e.g., "CalculatorAPI")
        module_path: Python module path where the class is defined (e.g., "my_tools.calculator")
        stateless: Whether the class is stateless (doesn't need _load_scenario)

    Example:
        register_tool_class("CalculatorAPI", "my_tools.calculator")
        register_tool_class("MathUtils", "utils.math", stateless=True)
    """
    global _CUSTOM_TOOL_REGISTRY, _CUSTOM_STATELESS_CLASSES

    _CUSTOM_TOOL_REGISTRY[class_name] = module_path

    if stateless and class_name not in _CUSTOM_STATELESS_CLASSES:
        _CUSTOM_STATELESS_CLASSES.append(class_name)
    elif not stateless and class_name in _CUSTOM_STATELESS_CLASSES:
        _CUSTOM_STATELESS_CLASSES.remove(class_name)

    LOG.info(f"Registered custom tool class: {class_name} -> {module_path} (stateless={stateless})")


def register_multiple_tools(tools_dict: Dict[str, str], stateless_classes: Optional[List[str]] = None) -> None:
    """
    Register multiple tool classes at once.

    Args:
        tools_dict: Dictionary mapping class names to module paths
        stateless_classes: List of class names that are stateless

    Example:
        register_multiple_tools({
            "CalculatorAPI": "my_tools.calculator",
            "FileSystemAPI": "my_tools.filesystem"
        }, stateless_classes=["MathUtils"])
    """
    for class_name, module_path in tools_dict.items():
        stateless = stateless_classes and class_name in stateless_classes
        register_tool_class(class_name, module_path, stateless=stateless)


def _load_tools_from_env() -> None:
    """Load custom tools from environment variable."""
    env_tools = os.getenv("BFCL_CUSTOM_TOOLS")
    if not env_tools:
        return

    try:
        for tool_spec in env_tools.split(","):
            tool_spec = tool_spec.strip()
            if ":" not in tool_spec:
                LOG.warning(f"Invalid tool specification in BFCL_CUSTOM_TOOLS: {tool_spec}")
                continue

            class_name, module_path = tool_spec.split(":", 1)
            register_tool_class(class_name.strip(), module_path.strip())

        LOG.info(f"Loaded {len(env_tools.split(','))} tools from BFCL_CUSTOM_TOOLS environment variable")

    except Exception as e:
        LOG.error(f"Error loading tools from environment variable: {e}")


def _load_tools_from_config() -> None:
    """Load custom tools from configuration file."""
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
        return

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        tools_dict = config.get("tools", {})
        stateless_classes = config.get("stateless_classes", [])

        register_multiple_tools(tools_dict, stateless_classes)
        LOG.info(f"Loaded {len(tools_dict)} tools from config file: {config_file}")

    except Exception as e:
        LOG.error(f"Error loading tools from config file {config_file}: {e}")


def _initialize_registry() -> None:
    """Initialize the registry by loading from all sources."""
    global _REGISTRY_INITIALIZED

    if _REGISTRY_INITIALIZED:
        return

    # Load from config file first
    _load_tools_from_config()

    # Load from environment (can override config)
    _load_tools_from_env()

    _REGISTRY_INITIALIZED = True


def get_custom_tool_mapping() -> Dict[str, str]:
    """
    Get the current custom tool mapping.

    Returns:
        Dictionary mapping class names to module paths
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
            "WeatherAPI": "weather_tools.api"
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


def list_registered_tools() -> None:
    """Print all currently registered tools."""
    _initialize_registry()

    print("🔧 BFCL Registered Custom Tools:")
    print("=" * 40)

    if not _CUSTOM_TOOL_REGISTRY:
        print("No custom tools registered.")
        print("\nTo register tools:")
        print("1. Set BFCL_CUSTOM_TOOLS env var: 'MyAPI:my_module'")
        print("2. Create ~/.bfcl_tools.json config file")
        print("3. Use register_tool_class() programmatically")
        return

    for class_name, module_path in _CUSTOM_TOOL_REGISTRY.items():
        stateless_marker = " (stateless)" if class_name in _CUSTOM_STATELESS_CLASSES else ""
        print(f"  {class_name} -> {module_path}{stateless_marker}")

    print(f"\nTotal: {len(_CUSTOM_TOOL_REGISTRY)} custom tools registered")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        output_file = sys.argv[2] if len(sys.argv) > 2 else "bfcl_tools.json"
        create_tools_config(output_file)
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        list_registered_tools()
    else:
        print("BFCL Tools Registry")
        print("Usage:")
        print("  python -m nemo_skills.inference.eval.bfcl_registry create-config [filename]")
        print("  python -m nemo_skills.inference.eval.bfcl_registry list")
