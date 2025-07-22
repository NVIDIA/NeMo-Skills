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
import textwrap
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from nemo_skills.inference.eval.bfcl_registry import (
    get_custom_tool_mapping,
    get_custom_stateless_classes,
    get_tool_class,
    list_registered_tools,
    create_tools_config,
    validate_tools_config,
    _initialize_registry,
    _load_tools_from_config,
    _is_file_path,
    _load_module_from_file,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry state before each test."""
    import nemo_skills.inference.eval.bfcl_registry as registry_module

    # Store original state
    original_registry = registry_module._CUSTOM_TOOL_REGISTRY.copy()
    original_stateless = registry_module._CUSTOM_STATELESS_CLASSES.copy()
    original_loaded_modules = registry_module._LOADED_MODULES.copy()
    original_initialized = registry_module._REGISTRY_INITIALIZED

    # Clear registry
    registry_module._CUSTOM_TOOL_REGISTRY.clear()
    registry_module._CUSTOM_STATELESS_CLASSES.clear()
    registry_module._LOADED_MODULES.clear()
    registry_module._REGISTRY_INITIALIZED = False

    yield

    # Restore original state
    registry_module._CUSTOM_TOOL_REGISTRY.clear()
    registry_module._CUSTOM_TOOL_REGISTRY.update(original_registry)
    registry_module._CUSTOM_STATELESS_CLASSES.clear()
    registry_module._CUSTOM_STATELESS_CLASSES.extend(original_stateless)
    registry_module._LOADED_MODULES.clear()
    registry_module._LOADED_MODULES.update(original_loaded_modules)
    registry_module._REGISTRY_INITIALIZED = original_initialized


class TestPathDetection:
    """Test file path vs module path detection."""

    def test_is_file_path_absolute(self):
        """Test detection of absolute file paths."""
        assert _is_file_path("/absolute/path/to/file.py") is True
        assert _is_file_path("/home/user/module.py") is True
        assert _is_file_path("C:\\Windows\\module.py") is True

    def test_is_file_path_python_extension(self):
        """Test detection of .py files."""
        assert _is_file_path("relative_file.py") is True
        assert _is_file_path("../parent/file.py") is True

    def test_is_module_path(self):
        """Test detection of module paths."""
        assert _is_file_path("my_module") is False
        assert _is_file_path("my_package.submodule") is False
        assert _is_file_path("package.sub.module") is False


class TestModuleLoading:
    """Test dynamic module loading from files."""

    def test_load_module_from_file_success(self):
        """Test successful loading of a module from a file."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(textwrap.dedent("""
                class TestAPI:
                    def __init__(self):
                        self.name = "TestAPI"

                    def method(self):
                        return "test_result"
            """))
            temp_file = f.name

        try:
            module = _load_module_from_file(temp_file, "test_module")
            assert hasattr(module, "TestAPI")

            # Test the class
            test_class = getattr(module, "TestAPI")
            instance = test_class()
            assert instance.name == "TestAPI"
            assert instance.method() == "test_result"

        finally:
            os.unlink(temp_file)

    def test_load_module_from_file_caching(self):
        """Test that modules are cached correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class CachedAPI: pass")
            temp_file = f.name

        try:
            module1 = _load_module_from_file(temp_file, "cached_module")
            module2 = _load_module_from_file(temp_file, "cached_module")

            # Should be the same object due to caching
            assert module1 is module2

        finally:
            os.unlink(temp_file)

    def test_load_module_from_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError, match="Python file not found"):
            _load_module_from_file("/nonexistent/file.py", "test_module")

    def test_load_module_from_file_invalid_python(self):
        """Test handling of invalid Python syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax !!!")
            temp_file = f.name

        try:
            with pytest.raises(Exception):  # Should raise some kind of syntax/execution error
                _load_module_from_file(temp_file, "invalid_module")
        finally:
            os.unlink(temp_file)


class TestGetToolClass:
    """Test the get_tool_class function."""

    def test_get_tool_class_from_file(self):
        """Test loading a tool class from a file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(textwrap.dedent("""
                class FileAPI:
                    def __init__(self):
                        self.source = "file"

                    def get_source(self):
                        return self.source
            """))
            temp_file = f.name

        try:
            tool_class = get_tool_class("FileAPI", temp_file)
            assert tool_class.__name__ == "FileAPI"

            # Test instantiation
            instance = tool_class()
            assert instance.get_source() == "file"

        finally:
            os.unlink(temp_file)

    def test_get_tool_class_from_module(self):
        """Test loading a tool class from a module path."""
        # Mock importlib.import_module
        mock_module = Mock()
        mock_class = Mock()
        mock_class.__name__ = "ModuleAPI"
        mock_module.ModuleAPI = mock_class

        with patch('importlib.import_module', return_value=mock_module):
            tool_class = get_tool_class("ModuleAPI", "test.module")
            assert tool_class is mock_class

    def test_get_tool_class_missing_class_in_file(self):
        """Test error when class is missing from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class WrongAPI: pass")
            temp_file = f.name

        try:
            with pytest.raises(AttributeError, match="Class 'MissingAPI' not found"):
                get_tool_class("MissingAPI", temp_file)
        finally:
            os.unlink(temp_file)

    def test_get_tool_class_missing_class_in_module(self):
        """Test error when class is missing from module."""
        mock_module = Mock()
        del mock_module.MissingAPI  # Ensure attribute doesn't exist

        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(AttributeError, match="Class 'MissingAPI' not found"):
                get_tool_class("MissingAPI", "test.module")


class TestConfigFileLoading:
    """Test JSON configuration file loading."""

    def test_load_from_config_file_basic(self):
        """Test loading basic configuration from JSON file."""
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
                assert mapping["ConfigAPI"] == "config.module"
                assert "AnotherAPI" in mapping
                assert mapping["AnotherAPI"] == "another.module"

                stateless = get_custom_stateless_classes()
                assert "ConfigAPI" in stateless
                assert "AnotherAPI" not in stateless
        finally:
            os.unlink(config_file)

    def test_load_from_config_file_with_file_paths(self):
        """Test loading configuration with file paths."""
        config_data = {
            "tools": {
                "ModuleAPI": "my.module",
                "FileAPI": "/path/to/file.py",
                "RelativeAPI": "relative.py"
            },
            "stateless_classes": ["FileAPI"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert mapping["ModuleAPI"] == "my.module"
                assert mapping["FileAPI"] == "/path/to/file.py"
                assert mapping["RelativeAPI"] == "relative.py"

                stateless = get_custom_stateless_classes()
                assert "FileAPI" in stateless
        finally:
            os.unlink(config_file)

    def test_config_file_search_priority(self):
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
            assert mapping["PriorityAPI"] == "priority.module"
        finally:
            if temp_config.exists():
                temp_config.unlink()

    def test_invalid_config_file_handled(self):
        """Test that invalid configuration files are handled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                with pytest.raises(Exception):  # Should raise due to invalid JSON
                    _load_tools_from_config()
        finally:
            os.unlink(config_file)

    def test_missing_config_file_handled(self):
        """Test that missing configuration files are handled gracefully."""
        with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': '/nonexistent/config.json'}):
            _load_tools_from_config()  # Should not raise exception

            mapping = get_custom_tool_mapping()
            assert len(mapping) == 0

    def test_nonexistent_file_path_warning(self, caplog):
        """Test warning when file paths don't exist."""
        config_data = {
            "tools": {
                "FileAPI": "/nonexistent/file.py"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                # Should still load but with warning
                mapping = get_custom_tool_mapping()
                assert "FileAPI" in mapping

                # Check for warning in logs
                assert any("does not exist" in record.message for record in caplog.records)
        finally:
            os.unlink(config_file)


class TestRegistryInitialization:
    """Test registry initialization."""

    def test_initialization_loads_config(self):
        """Test that initialization loads configuration."""
        config_data = {"tools": {"InitAPI": "init.module"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _initialize_registry()

                mapping = get_custom_tool_mapping()
                assert "InitAPI" in mapping
                assert mapping["InitAPI"] == "init.module"
        finally:
            os.unlink(config_file)

    def test_initialization_idempotent(self):
        """Test that initialization is idempotent."""
        config_data = {"tools": {"IdempotentAPI": "idempotent.module"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _initialize_registry()
                first_mapping = get_custom_tool_mapping()

                _initialize_registry()  # Should not reload
                second_mapping = get_custom_tool_mapping()

                assert first_mapping == second_mapping
        finally:
            os.unlink(config_file)


class TestValidateToolsConfig:
    """Test configuration validation."""

    def test_validate_config_success(self):
        """Test successful validation of all tools."""
        # Create temporary Python files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
            f1.write("class ValidAPI1: pass")
            temp_file1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
            f2.write("class ValidAPI2: pass")
            temp_file2 = f2.name

        config_data = {
            "tools": {
                "ValidAPI1": temp_file1,
                "ValidAPI2": temp_file2
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            result = validate_tools_config(config_file)
            assert result is True
        finally:
            os.unlink(config_file)
            os.unlink(temp_file1)
            os.unlink(temp_file2)

    def test_validate_config_failure(self):
        """Test validation failure with invalid tools."""
        config_data = {
            "tools": {
                "MissingAPI": "/nonexistent/file.py"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            result = validate_tools_config(config_file)
            assert result is False
        finally:
            os.unlink(config_file)

    def test_validate_config_mixed_results(self):
        """Test validation with mixed valid/invalid tools."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class ValidAPI: pass")
            valid_file = f.name

        config_data = {
            "tools": {
                "ValidAPI": valid_file,
                "InvalidAPI": "/nonexistent/file.py"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            result = validate_tools_config(config_file)
            assert result is False  # Should fail if any tool fails
        finally:
            os.unlink(config_file)
            os.unlink(valid_file)


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

            # Check for example file path
            assert any("/nemo_run/code/custom_api.py" in path for path in config["tools"].values())

        finally:
            if Path(output_file).exists():
                os.unlink(output_file)

    def test_list_registered_tools_empty(self, capsys):
        """Test listing when no tools are registered."""
        list_registered_tools()
        captured = capsys.readouterr()
        assert "No custom tools registered" in captured.out
        assert "create a JSON config file" in captured.out

    def test_list_registered_tools_with_tools(self, capsys):
        """Test listing registered tools."""
        config_data = {
            "tools": {
                "ModuleAPI": "list.module",
                "FileAPI": "/path/to/file.py",
                "StatelessAPI": "stateless.module"
            },
            "stateless_classes": ["StatelessAPI"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                list_registered_tools()
                captured = capsys.readouterr()

                assert "ModuleAPI -> list.module [module]" in captured.out
                assert "FileAPI -> /path/to/file.py [file]" in captured.out
                assert "StatelessAPI -> stateless.module [module] (stateless)" in captured.out
                assert "Total: 3 custom tools registered" in captured.out
        finally:
            os.unlink(config_file)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_config_file(self):
        """Test handling of empty configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert len(mapping) == 0
        finally:
            os.unlink(config_file)

    def test_config_without_tools_section(self):
        """Test configuration without tools section."""
        config_data = {"other_section": "value"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert len(mapping) == 0
        finally:
            os.unlink(config_file)

    def test_config_without_stateless_section(self):
        """Test configuration without stateless_classes section."""
        config_data = {"tools": {"TestAPI": "test.module"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert "TestAPI" in mapping

                stateless = get_custom_stateless_classes()
                assert len(stateless) == 0
        finally:
            os.unlink(config_file)

    def test_registry_isolation(self):
        """Test that each test has isolated registry state."""
        config_data = {"tools": {"IsolationAPI": "isolation.module"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'BFCL_TOOLS_CONFIG': config_file}):
                _load_tools_from_config()

                mapping = get_custom_tool_mapping()
                assert "IsolationAPI" in mapping
        finally:
            os.unlink(config_file)

        # This should be isolated in the next test due to the reset_registry fixture
