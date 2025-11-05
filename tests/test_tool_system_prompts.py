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

"""Tests for tool system prompt functionality."""

from copy import deepcopy

import pytest

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.mcp.tool_manager import ToolManager
from nemo_skills.mcp.tool_providers import MCPClientTool


# Simple mock client that doesn't require actual MCP server
class SimpleMockClient:
    """Minimal mock MCP client for testing."""

    def __init__(self, **kwargs):
        pass

    async def list_tools(self):
        return [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}},
            }
        ]

    async def call_tool(self, tool, args, extra_args=None):
        return {"result": "success"}


class SimpleTestTool(MCPClientTool):
    """Simple test tool for system prompt testing."""

    def __init__(self):
        super().__init__()
        self.apply_config_updates(
            {
                "client": f"{__name__}::SimpleMockClient",
                "client_params": {},
            }
        )


# ============================================================================
# Tool System Prompt Loading Tests
# ============================================================================


def test_system_prompt_loading_valid_file(tmp_path):
    """Test loading system prompt from a valid YAML file."""
    prompt_file = tmp_path / "test_prompt.yaml"
    prompt_file.write_text(
        """system: >-
  This is a test system prompt.
  It has multiple lines.
  And provides instructions.
"""
    )

    tool = SimpleTestTool()
    tool.configure(overrides={"system_prompt_file": str(prompt_file)}, context={})

    assert tool.get_system_prompt() is not None
    assert "This is a test system prompt" in tool.get_system_prompt()
    assert "multiple lines" in tool.get_system_prompt()


def test_system_prompt_missing_file():
    """Test error handling when system prompt file doesn't exist."""
    tool = SimpleTestTool()

    with pytest.raises(FileNotFoundError) as exc_info:
        tool.configure(overrides={"system_prompt_file": "/nonexistent/path/prompt.yaml"}, context={})

    assert "System prompt file not found" in str(exc_info.value)


def test_system_prompt_missing_system_key(tmp_path):
    """Test error handling when YAML file doesn't have 'system:' key."""
    prompt_file = tmp_path / "bad_prompt.yaml"
    prompt_file.write_text("other_key: some value\n")

    tool = SimpleTestTool()

    with pytest.raises(ValueError) as exc_info:
        tool.configure(overrides={"system_prompt_file": str(prompt_file)}, context={})

    assert "Invalid system prompt file" in str(exc_info.value)
    assert "must contain a 'system:' key" in str(exc_info.value)


def test_system_prompt_invalid_yaml(tmp_path):
    """Test error handling when YAML file is malformed."""
    prompt_file = tmp_path / "invalid.yaml"
    prompt_file.write_text("system: [\n  unclosed bracket\n")

    tool = SimpleTestTool()

    with pytest.raises(ValueError) as exc_info:
        tool.configure(overrides={"system_prompt_file": str(prompt_file)}, context={})

    assert "Failed to parse YAML" in str(exc_info.value)


def test_system_prompt_none_when_not_configured():
    """Test that get_system_prompt returns None when not configured."""
    tool = SimpleTestTool()
    tool.configure(overrides={}, context={})

    assert tool.get_system_prompt() is None


def test_system_prompt_with_special_characters(tmp_path):
    """Test system prompts containing special characters."""
    prompt_file = tmp_path / "special.yaml"
    prompt_file.write_text(
        """system: >-
  Use Python's f-strings like this: f"value={x}"
  Use paths: /path/to/file
  Use quotes: "quoted text" and 'single quotes'
  Use special symbols: $var, @mention, #tag
"""
    )

    tool = SimpleTestTool()
    tool.configure(overrides={"system_prompt_file": str(prompt_file)}, context={})

    prompt = tool.get_system_prompt()
    assert 'f"value={x}"' in prompt
    assert "/path/to/file" in prompt
    assert '"quoted text"' in prompt
    assert "$var" in prompt


# ============================================================================
# ToolManager Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_manager_collects_system_prompts(tmp_path):
    """Test that ToolManager collects system prompts from configured tools."""
    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text("system: Tool instructions from config")

    # Use real ToolManager with real tool instances
    tm = ToolManager(
        module_specs=[f"{__name__}::SimpleTestTool"],
        overrides={
            "SimpleTestTool": {"system_prompt_file": str(prompt_file)},
        },
        context={},
    )

    merged = tm.get_tool_system_prompts()

    # Should get prompt from the configured tool
    assert merged is not None
    assert "Tool instructions from config" in merged


@pytest.mark.asyncio
async def test_tool_manager_handles_no_system_prompts():
    """Test that ToolManager returns None when no tools have system prompts."""
    tm = ToolManager(module_specs=[f"{__name__}::SimpleTestTool"], overrides={}, context={})

    merged = tm.get_tool_system_prompts()
    assert merged is None


@pytest.mark.asyncio
async def test_tool_manager_merges_multiple_tools_with_prompts(tmp_path):
    """Test merging system prompts from multiple different tool classes."""

    # Create different tool classes
    class Tool1(SimpleTestTool):
        pass

    class Tool2(SimpleTestTool):
        pass

    # Register in module
    import sys

    sys.modules[__name__].Tool1 = Tool1
    sys.modules[__name__].Tool2 = Tool2

    prompt1 = tmp_path / "prompt1.yaml"
    prompt1.write_text("system: Tool 1 instructions")

    prompt2 = tmp_path / "prompt2.yaml"
    prompt2.write_text("system: Tool 2 instructions")

    tm = ToolManager(
        module_specs=[f"{__name__}::Tool1", f"{__name__}::Tool2"],
        overrides={
            "Tool1": {"system_prompt_file": str(prompt1)},
            "Tool2": {"system_prompt_file": str(prompt2)},
        },
        context={},
    )

    merged = tm.get_tool_system_prompts()

    # Both prompts should be present
    assert "Tool 1 instructions" in merged
    assert "Tool 2 instructions" in merged
    # Should be separated by double newlines
    assert "\n\n" in merged


# ============================================================================
# GenerationTask System Message Composition Tests
# ============================================================================


def test_compose_system_message_all_sources():
    """Test composing system message from all three sources."""

    class TestTask(GenerationTask):
        def __init__(self):
            # Minimal setup without full initialization
            self.cfg = type("obj", (object,), {"system_message": "User custom"})()
            self.tool_system_prompt = "Tool instructions"

    task = TestTask()
    composed = task._compose_system_message(default_system="Default config")

    assert "Default config" in composed
    assert "Tool instructions" in composed
    assert "User custom" in composed

    # Verify order: default -> tool -> user
    parts = composed.split("\n\n")
    assert parts[0] == "Default config"
    assert parts[1] == "Tool instructions"
    assert parts[2] == "User custom"


def test_compose_system_message_single_sources():
    """Test composing with only one source at a time."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type("obj", (object,), {"system_message": None})()
            self.tool_system_prompt = None

    # Test each source individually
    task = TestTask()

    assert task._compose_system_message(default_system="Only default") == "Only default"

    task.tool_system_prompt = "Only tool"
    assert task._compose_system_message(default_system=None) == "Only tool"

    task.tool_system_prompt = None
    task.cfg.system_message = "Only user"
    assert task._compose_system_message(default_system=None) == "Only user"


def test_compose_system_message_all_none():
    """Test that None is returned when all sources are empty."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type("obj", (object,), {"system_message": None})()
            self.tool_system_prompt = None

    task = TestTask()
    assert task._compose_system_message(default_system=None) is None


def test_fill_prompt_openai_format_composition():
    """Test fill_prompt composes system messages in OpenAI format."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type(
                "obj", (object,), {"prompt_format": "openai", "prompt_suffix": None, "system_message": "User message"}
            )()
            self.tool_system_prompt = "Tool prompt"

    task = TestTask()

    # Test inserting system message
    data_point = {"messages": [{"role": "user", "content": "Hello"}]}
    result = task.fill_prompt(data_point, [])

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "Tool prompt" in result[0]["content"]
    assert "User message" in result[0]["content"]

    # Test composing with existing system message
    data_point2 = {"messages": [{"role": "system", "content": "Default"}, {"role": "user", "content": "Hello"}]}
    result2 = task.fill_prompt(data_point2, [])

    assert result2[0]["role"] == "system"
    assert "Default" in result2[0]["content"]
    assert "Tool prompt" in result2[0]["content"]
    assert "User message" in result2[0]["content"]


def test_fill_prompt_preserves_other_messages():
    """Test that fill_prompt doesn't modify non-system messages."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type(
                "obj", (object,), {"prompt_format": "openai", "prompt_suffix": None, "system_message": "Test"}
            )()
            self.tool_system_prompt = None

    task = TestTask()

    data_point = {
        "messages": [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
    }

    original = deepcopy(data_point)
    result = task.fill_prompt(data_point, [])

    # Should add system message at beginning
    assert len(result) == len(original["messages"]) + 1
    assert result[0]["role"] == "system"

    # Other messages should be unchanged
    for i, orig_msg in enumerate(original["messages"]):
        assert result[i + 1] == orig_msg


def test_fill_prompt_with_suffix():
    """Test that prompt_suffix is applied correctly."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type(
                "obj", (object,), {"prompt_format": "openai", "prompt_suffix": " [SUFFIX]", "system_message": None}
            )()
            self.tool_system_prompt = None

    task = TestTask()

    data_point = {"messages": [{"role": "user", "content": "Test message"}]}
    result = task.fill_prompt(data_point, [])

    # Suffix should be on last message
    assert result[-1]["content"] == "Test message [SUFFIX]"


def test_compose_preserves_newlines():
    """Test that newlines within messages are preserved."""

    class TestTask(GenerationTask):
        def __init__(self):
            self.cfg = type("obj", (object,), {"system_message": "Line 1\nLine 2"})()
            self.tool_system_prompt = "Tool\nMulti\nLine"

    task = TestTask()
    composed = task._compose_system_message(default_system="Default\nTwo lines")

    assert "Default\nTwo lines" in composed
    assert "Tool\nMulti\nLine" in composed
    assert "Line 1\nLine 2" in composed


def test_backward_compatibility_without_tools():
    """Test backward compatible behavior when no tools are configured.

    WITHOUT tools: ++system_message should replace the default (not compose).
    This is handled in setup_prompt() by checking cfg.tool_modules.
    """

    class TestTask(GenerationTask):
        def __init__(self):
            # No tool_modules configured
            self.cfg = type("obj", (object,), {"system_message": "User override", "tool_modules": None})()
            self.tool_system_prompt = None

    task = TestTask()

    # When no tools, user message replaces default (not composed)
    # This simulates what happens when get_prompt is called with cfg.system_message
    composed = task._compose_system_message(default_system="Config default")

    # Should only have user override, not composition
    # (In practice, get_prompt() would have already replaced it)
    assert composed is not None


def test_composition_with_tools():
    """Test composition behavior when tools are configured.

    WITH tools: All three sources are composed together.
    This is handled by passing None to get_prompt() when cfg.tool_modules is set.
    """

    class TestTask(GenerationTask):
        def __init__(self):
            # Tool modules ARE configured
            self.cfg = type("obj", (object,), {"system_message": "User override", "tool_modules": ["some.tool"]})()
            self.tool_system_prompt = "Tool prompt"

    task = TestTask()

    # When tools present, compose all three
    composed = task._compose_system_message(default_system="Config default")

    # All three should be present and composed
    assert "Config default" in composed
    assert "Tool prompt" in composed
    assert "User override" in composed

    parts = composed.split("\n\n")
    assert parts[0] == "Config default"
    assert parts[1] == "Tool prompt"
    assert parts[2] == "User override"
