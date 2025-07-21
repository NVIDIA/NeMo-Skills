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

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from nemo_skills.inference.eval.bfcl_registry import (
    register_tool_class,
    register_multiple_tools,
    get_custom_tool_mapping,
    get_custom_stateless_classes,
    list_registered_tools,
    create_tools_config,
    _initialize_registry,
    _load_tools_from_env,
    _load_tools_from_config,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry state before each test."""
    import nemo_skills.inference.eval.bfcl_registry as registry_module

    # Store original state
    original_registry = registry_module._CUSTOM_TOOL_REGISTRY.copy()
    original_stateless = registry_module._CUSTOM_STATELESS_CLASSES.copy()
    original_initialized = registry_module._REGISTRY_INITIALIZED

    # Clear registry
    registry_module._CUSTOM_TOOL_REGISTRY.clear()
    registry_module._CUSTOM_STATELESS_CLASSES.clear()
    registry_module._REGISTRY_INITIALIZED = False

    yield

    # Restore original state
    registry_module._CUSTOM_TOOL_REGISTRY.clear()
    registry_module._CUSTOM_TOOL_REGISTRY.update(original_registry)
    registry_module._CUSTOM_STATELESS_CLASSES.clear()
    registry_module._CUSTOM_STATELESS_CLASSES.extend(original_stateless)
    registry_module._REGISTRY_INITIALIZED = original_initialized


class TestProgrammaticRegistration:
    """Test programmatic tool registration."""

    def test_register_single_tool_class(self):
        """Test registering a single tool class."""
        register_tool_class("TestAPI", "test.module")

        mapping = get_custom_tool_mapping()
        assert "TestAPI" in mapping
        assert mapping["TestAPI"] == "test.module"

        stateless = get_custom_stateless_classes()
        assert "TestAPI" not in stateless

    def test_register_stateless_tool_class(self):
        """Test registering a stateless tool class."""
        register_tool_class("StatelessAPI", "test.stateless", stateless=True)

        mapping = get_custom_tool_mapping()
        assert "StatelessAPI" in mapping
        assert mapping["StatelessAPI"] == "test.stateless"

        stateless = get_custom_stateless_classes()
        assert "StatelessAPI" in stateless

    def test_register_multiple_tools(self):
        """Test registering multiple tools at once."""
        tools_dict = {
            "CalculatorAPI": "calc.module",
            "DatabaseAPI": "db.module",
            "FileAPI": "file.module"
        }
        stateless_classes = ["FileAPI"]

        register_multiple_tools(tools_dict, stateless_classes)

        mapping = get_custom_tool_mapping()
        for name, module in tools_dict.items():
            assert name in mapping
            assert mapping[name] == module

        stateless = get_custom_stateless_classes()
        assert "FileAPI" in stateless
        assert "CalculatorAPI" not in stateless
        assert "DatabaseAPI" not in stateless

    def test_override_existing_tool(self):
        """Test that later registrations override earlier ones."""
        register_tool_class("TestAPI", "old.module")
        register_tool_class("TestAPI", "new.module", stateless=True)

        mapping = get_custom_tool_mapping()
        assert mapping["TestAPI"] == "new.module"

        stateless = get_custom_stateless_classes()
        assert "TestAPI" in stateless

    def test_change_stateless_status(self):
        """Test changing the stateless status of a tool."""
        # Register as stateless first
        register_tool_class("TestAPI", "test.module", stateless=True)
        stateless = get_custom_stateless_classes()
        assert "TestAPI" in stateless

        # Re-register as stateful
        register_tool_class("TestAPI", "test.module", stateless=False)
        stateless = get_custom_stateless_classes()
        assert "TestAPI" not in stateless


class TestEnvironmentVariableRegistration:
    """Test environment variable-based tool registration."""

    @patch.dict(os.environ, {'BFCL_CUSTOM_TOOLS': 'TestAPI:test.module'})
    def test_single_tool_from_env(self):
        """Test loading a single tool from environment variable."""
        _load_tools_from_env()

        mapping = get_custom_tool_mapping()
        assert "TestAPI" in mapping
        assert mapping["TestAPI"] == "test.module"

    @patch.dict(os.environ, {'BFCL_CUSTOM_TOOLS': 'API1:module1,API2:module2,API3:module3'})
    def test_multiple_tools_from_env(self):
        """Test loading multiple tools from environment variable."""
        _load_tools_from_env()

        mapping = get_custom_tool_mapping()
        assert "API1" in mapping and mapping["API1"] == "module1"
        assert "API2" in mapping and mapping["API2"] == "module2"
        assert "API3" in mapping and mapping["API3"] == "module3"

    @patch.dict(os.environ, {'BFCL_CUSTOM_TOOLS': 'ValidAPI:module,InvalidSpec,AnotherAPI:another'})
    def test_invalid_env_specs_ignored(self):
        """Test that invalid environment specifications are ignored."""
        _load_tools_from_env()

        mapping = get_custom_tool_mapping()
        assert "ValidAPI" in mapping
        assert "AnotherAPI" in mapping
        assert "InvalidSpec" not in mapping

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_variable(self):
        """Test that missing environment variable is handled gracefully."""
        _load_tools_from_env()

        mapping = get_custom_tool_mapping()
        assert len(mapping) == 0


class TestConfigFileRegistration:
    """Test configuration file-based tool registration."""

    def test_load_from_config_file(self):
        """Test loading tools from a configuration file."""
        config_data = {
            "tools": {
                "ConfigAPI": "config.module",
                "AnotherAPI": "another.module"
            },
            "stateless_classes": ["ConfigAPI"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert "ConfigAPI" in mapping
                assert "AnotherAPI" in mapping

                stateless = get_custom_stateless_classes()
                assert "ConfigAPI" in stateless
                assert "AnotherAPI" not in stateless
        finally:
            os.unlink(config_file)

    def test_config_file_priority(self):
        """Test configuration file search priority."""
        config_data = {"tools": {"PriorityAPI": "priority.module"}}

        # Create temporary config in current directory
        temp_config = Path("bfcl_tools.json")
        with temp_config.open('w') as f:
            json.dump(config_data, f)

        try:
            _load_tools_from_config()

            mapping = get_custom_tool_mapping()
            assert "PriorityAPI" in mapping
        finally:
            if temp_config.exists():
                temp_config.unlink()

    def test_invalid_config_file_handled(self):
        """Test that invalid configuration files are handled gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()  # Should not raise exception

                mapping = get_custom_tool_mapping()
                assert len(mapping) == 0
        finally:
            os.unlink(config_file)

    def test_missing_config_file_handled(self):
        """Test that missing configuration files are handled gracefully."""
        with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': '/nonexistent/config.json'}):
            _load_tools_from_config()  # Should not raise exception

            mapping = get_custom_tool_mapping()
            assert len(mapping) == 0


class TestRegistryInitialization:
    """Test registry initialization and priority."""

    def test_initialization_priority(self):
        """Test that initialization loads config first, then environment."""
        config_data = {"tools": {"ConfigAPI": "config.module"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {
                'BFCL_TOOLS_CONFIG': config_file,
                'BFCL_CUSTOM_TOOLS': 'EnvAPI:env.module,ConfigAPI:env.override'
            }):
                _initialize_registry()

                mapping = get_custom_tool_mapping()
                # Environment should override config
                assert mapping["ConfigAPI"] == "env.override"
                assert mapping["EnvAPI"] == "env.module"
        finally:
            os.unlink(config_file)

    def test_initialization_idempotent(self):
        """Test that initialization is idempotent."""
        register_tool_class("TestAPI", "test.module")

        _initialize_registry()
        first_mapping = get_custom_tool_mapping()

        _initialize_registry()
        second_mapping = get_custom_tool_mapping()

        assert first_mapping == second_mapping


class TestBFCLUtilsIntegration:
    """Test integration with BFCL utils."""

    def test_integration_with_bfcl_utils(self):
        """Test that registered tools appear in BFCL utils mappings."""
        # Register some tools
        register_tool_class("IntegrationAPI", "integration.module")
        register_tool_class("StatelessAPI", "stateless.module", stateless=True)

        # Import BFCL utils to trigger integration
        from nemo_skills.inference.eval.bfcl_utils import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES

        # Check that custom tools appear in the mappings
        assert "IntegrationAPI" in CLASS_FILE_PATH_MAPPING
        assert CLASS_FILE_PATH_MAPPING["IntegrationAPI"] == "integration.module"

        assert "StatelessAPI" in CLASS_FILE_PATH_MAPPING
        assert "StatelessAPI" in STATELESS_CLASSES
        assert "IntegrationAPI" not in STATELESS_CLASSES

        # Check that base tools are still present
        assert "MathAPI" in CLASS_FILE_PATH_MAPPING
        assert "GorillaFileSystem" in CLASS_FILE_PATH_MAPPING


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_tools_config(self):
        """Test creating a tools configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            os.unlink(output_file)  # Remove the file so create_tools_config can create it
            create_tools_config(output_file)

            assert Path(output_file).exists()

            with open(output_file, 'r') as f:
                config = json.load(f)

            assert "tools" in config
            assert "stateless_classes" in config
            assert isinstance(config["tools"], dict)
            assert isinstance(config["stateless_classes"], list)

        finally:
            if Path(output_file).exists():
                os.unlink(output_file)

    def test_list_registered_tools(self, capsys):
        """Test listing registered tools."""
        # Test with no tools registered
        list_registered_tools()
        captured = capsys.readouterr()
        assert "No custom tools registered" in captured.out

        # Register some tools and test again
        register_tool_class("ListTestAPI", "list.module")
        register_tool_class("StatelessListAPI", "stateless.list", stateless=True)

        list_registered_tools()
        captured = capsys.readouterr()
        assert "ListTestAPI -> list.module" in captured.out
        assert "StatelessListAPI -> stateless.list (stateless)" in captured.out
        assert "Total: 2 custom tools registered" in captured.out


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_environment_variable(self):
        """Test handling of empty environment variable."""
        with patch.dict(os.environ, {'BFCL_CUSTOM_TOOLS': ''}):
            _load_tools_from_env()

            mapping = get_custom_tool_mapping()
            assert len(mapping) == 0

    def test_whitespace_handling_in_env(self):
        """Test proper whitespace handling in environment variable."""
        with patch.dict(os.environ, {'BFCL_CUSTOM_TOOLS': ' API1 : module1 , API2:module2 '}):
            _load_tools_from_env()

            mapping = get_custom_tool_mapping()
            assert "API1" in mapping and mapping["API1"] == "module1"
            assert "API2" in mapping and mapping["API2"] == "module2"

    def test_duplicate_registrations(self):
        """Test handling of duplicate registrations."""
        register_tool_class("DuplicateAPI", "first.module")
        register_tool_class("DuplicateAPI", "second.module")

        mapping = get_custom_tool_mapping()
        # Should contain the last registration
        assert mapping["DuplicateAPI"] == "second.module"

    def test_registry_isolation(self):
        """Test that each test has isolated registry state."""
        register_tool_class("IsolationAPI", "isolation.module")

        mapping = get_custom_tool_mapping()
        assert "IsolationAPI" in mapping

        # This should be isolated in the next test due to the reset_registry fixture
