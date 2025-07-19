"""
Pytest tests for the Lean 4 LLM Tool.

Tests the comprehensive LLM tool interface designed for agents like Qwen3.
Covers all operation modes, configuration options, schema generation,
and integration scenarios.
"""

import pytest
import json
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nemo_skills.code_execution.lean4.llm_tool import (
    LeanLLMTool,
    ToolCapabilities,
    ToolResult,
    OperationType,
    create_basic_tool,
    create_interactive_tool,
    create_validation_tool,
    get_qwen3_tool_config,
)


class TestToolCapabilities:
    """Test ToolCapabilities configuration."""

    def test_default_capabilities(self):
        """Test default capabilities (all enabled)."""
        caps = ToolCapabilities()
        assert caps.execute == True
        assert caps.edit_clause == True
        assert caps.add_structure == True
        assert caps.validate == True
        assert caps.retrieve == True

    def test_custom_capabilities(self):
        """Test custom capability configuration."""
        caps = ToolCapabilities(
            execute=True,
            edit_clause=False,
            add_structure=False,
            validate=True,
            retrieve=False
        )
        assert caps.execute == True
        assert caps.edit_clause == False
        assert caps.add_structure == False
        assert caps.validate == True
        assert caps.retrieve == False


class TestToolResult:
    """Test ToolResult structure."""

    def test_tool_result_creation(self):
        """Test ToolResult creation and attributes."""
        result = ToolResult(
            success=True,
            operation="execute",
            result={"test": "data"},
            message="Test message",
            error=None
        )

        assert result.success == True
        assert result.operation == "execute"
        assert result.result == {"test": "data"}
        assert result.message == "Test message"
        assert result.error is None

    def test_tool_result_with_error(self):
        """Test ToolResult with error."""
        result = ToolResult(
            success=False,
            operation="test",
            result={},
            message="Error occurred",
            error="Test error"
        )

        assert result.success == False
        assert result.error == "Test error"


class TestLeanLLMTool:
    """Test the main LeanLLMTool class."""

    def test_initialization_default(self):
        """Test tool initialization with default settings."""
        tool = LeanLLMTool()

        assert tool.capabilities.execute == True
        assert tool.capabilities.edit_clause == True
        assert tool.mathlib_enabled == True
        assert tool.current_session is None
        assert len(tool.session_history) == 0

    def test_initialization_custom(self):
        """Test tool initialization with custom settings."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=True, retrieve=False)
        tool = LeanLLMTool(capabilities=caps, mathlib_enabled=False)

        assert tool.capabilities.execute == True
        assert tool.capabilities.edit_clause == False
        assert tool.mathlib_enabled == False

    def test_operation_enabled_check(self):
        """Test operation enablement checking."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=True, validate=False, retrieve=True)
        tool = LeanLLMTool(capabilities=caps)

        assert tool._is_operation_enabled("execute") == True
        assert tool._is_operation_enabled("edit_clause") == False
        assert tool._is_operation_enabled("add_structure") == True
        assert tool._is_operation_enabled("validate") == False
        assert tool._is_operation_enabled("retrieve") == True
        assert tool._is_operation_enabled("invalid_op") == False


class TestSchemaGeneration:
    """Test JSON schema generation."""

    def test_schema_all_capabilities(self):
        """Test schema generation with all capabilities enabled."""
        tool = LeanLLMTool()
        schema = tool.get_tool_schema()

        assert schema["name"] == "lean4_tool"
        assert "description" in schema
        assert "parameters" in schema

        # Check all operations are included
        operations = schema["parameters"]["properties"]["operation"]["enum"]
        assert "execute" in operations
        assert "edit_clause" in operations
        assert "add_structure" in operations
        assert "validate" in operations
        assert "retrieve" in operations

        # Check conditional schemas exist
        assert "allOf" in schema["parameters"]
        assert len(schema["parameters"]["allOf"]) == 5

    def test_schema_limited_capabilities(self):
        """Test schema generation with limited capabilities."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=True, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)
        schema = tool.get_tool_schema()

        # Check only enabled operations are included
        operations = schema["parameters"]["properties"]["operation"]["enum"]
        assert "execute" in operations
        assert "validate" in operations
        assert "edit_clause" not in operations
        assert "add_structure" not in operations
        assert "retrieve" not in operations

        # Check correct number of conditional schemas
        assert len(schema["parameters"]["allOf"]) == 2

    def test_schema_no_capabilities(self):
        """Test schema generation with no capabilities."""
        caps = ToolCapabilities(execute=False, edit_clause=False, add_structure=False, validate=False, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)
        schema = tool.get_tool_schema()

        # Should have empty operations list
        operations = schema["parameters"]["properties"]["operation"]["enum"]
        assert len(operations) == 0


class TestToolDescription:
    """Test dynamic tool description generation."""

    def test_description_all_capabilities(self):
        """Test description with all capabilities."""
        tool = LeanLLMTool()
        desc = tool.get_tool_description()

        assert "theorem prover tool" in desc.lower()
        assert "execute theorems" in desc
        assert "edit proof clauses" in desc
        assert "add new proof structure" in desc
        assert "validate with #check" in desc
        assert "retrieve current development" in desc

    def test_description_limited_capabilities(self):
        """Test description with limited capabilities."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=True, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)
        desc = tool.get_tool_description()

        assert "execute theorems" in desc
        assert "validate with #check" in desc
        assert "edit proof clauses" not in desc
        assert "add new proof structure" not in desc


class TestExecuteOperation:
    """Test the execute operation."""

    def test_execute_proof_success(self):
        """Test successful theorem execution."""
        tool = LeanLLMTool()

        # Mock successful proof result
        with patch.object(tool.prover, 'run') as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_result.proof_complete = True
            mock_result.has_sorry = False
            mock_result.response = "Success"
            mock_result.error = None
            mock_result.proof_state = None
            mock_run.return_value = mock_result

            result = tool(operation="execute", code="theorem test : True := trivial")

            assert result.success == True
            assert result.operation == "execute"
            assert result.result["success"] == True
            assert result.result["proof_complete"] == True
            assert "successful" in result.message.lower()

    def test_execute_proof_failure(self):
        """Test failed theorem execution."""
        tool = LeanLLMTool()

        # Mock failed proof result
        with patch.object(tool.prover, 'run') as mock_run:
            mock_result = Mock()
            mock_result.success = False
            mock_result.proof_complete = False
            mock_result.has_sorry = False
            mock_result.response = "[error] unknown tactic"
            mock_result.error = "unknown tactic"
            mock_result.proof_state = None
            mock_run.return_value = mock_result

            result = tool(operation="execute", code="theorem test : True := by invalid_tactic")

            assert result.success == False
            assert result.operation == "execute"
            assert result.result["success"] == False

    def test_execute_command_mode(self):
        """Test execution in command mode."""
        tool = LeanLLMTool()

        with patch.object(tool.prover, 'run_command') as mock_run_command:
            mock_result = Mock()
            mock_result.success = True
            mock_result.proof_complete = False
            mock_result.has_sorry = False
            mock_result.response = "Nat : Type"
            mock_result.error = None
            mock_result.proof_state = None
            mock_run_command.return_value = mock_result

            result = tool(operation="execute", code="#check Nat", mode="command")

            mock_run_command.assert_called_once_with("#check Nat")
            assert result.success == True

    def test_execute_disabled(self):
        """Test execute operation when disabled."""
        caps = ToolCapabilities(execute=False, edit_clause=False, add_structure=False, validate=True, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)

        result = tool(operation="execute", code="theorem test : True := trivial")

        assert result.success == False
        assert "not enabled" in result.message
        assert "disabled" in result.error


class TestEditClauseOperation:
    """Test the edit_clause operation."""

    def test_edit_clause_success(self):
        """Test successful clause editing."""
        tool = LeanLLMTool()

        # Mock agent methods
        with patch.object(tool.agent, 'load_theorem') as mock_load, \
             patch.object(tool.agent, 'edit_clause') as mock_edit:

            # Mock load result
            mock_load.return_value = {"success": True}

            # Mock edit result
            mock_edit.return_value = {
                "edit_successful": True,
                "clause_type": "sorry",
                "updated_code": "theorem test : True := trivial",
                "compilation_result": {
                    "success": True,
                    "has_errors": False,
                    "messages": [],
                    "editable_clauses": []
                }
            }

            result = tool(
                operation="edit_clause",
                theorem_code="theorem test : True := by sorry",
                clause_id="sorry_0",
                new_content="trivial"
            )

            assert result.success == True
            assert result.operation == "edit_clause"
            assert result.result["edit_successful"] == True
            assert result.result["clause_type"] == "sorry"

    def test_edit_clause_no_session(self):
        """Test clause editing without active session."""
        tool = LeanLLMTool()

        result = tool(operation="edit_clause", clause_id="sorry_0", new_content="trivial")

        assert result.success == False
        assert "no interactive session" in result.message.lower()

    def test_edit_clause_disabled(self):
        """Test edit_clause operation when disabled."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=False, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)

        result = tool(operation="edit_clause", clause_id="sorry_0", new_content="trivial")

        assert result.success == False
        assert "not enabled" in result.message


class TestAddStructureOperation:
    """Test the add_structure operation."""

    def test_add_structure_success(self):
        """Test successful structure addition."""
        tool = LeanLLMTool()

        with patch.object(tool.agent, 'load_theorem') as mock_load, \
             patch.object(tool.agent, 'add_proof_structure') as mock_add, \
             patch.object(tool.agent, 'get_interactive_panel') as mock_panel:

            mock_load.return_value = {"success": True}
            mock_add.return_value = {
                "edit_successful": True,
                "compilation_result": {"success": True, "has_errors": False, "messages": []}
            }
            mock_panel.return_value = {"current_code": "updated code"}

            result = tool(
                operation="add_structure",
                theorem_code="theorem test : P → P := by sorry",
                structure_lines=["intro h", "exact h"]
            )

            assert result.success == True
            assert result.operation == "add_structure"

    def test_add_structure_disabled(self):
        """Test add_structure operation when disabled."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=False, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)

        result = tool(operation="add_structure", structure_lines=["intro h"])

        assert result.success == False
        assert "not enabled" in result.message


class TestValidateOperation:
    """Test the validate operation."""

    def test_validate_success(self):
        """Test successful validation."""
        tool = LeanLLMTool()

        with patch.object(tool.prover, 'run_command') as mock_run_command:
            mock_result = Mock()
            mock_result.success = True
            mock_result.response = "Nat : Type"
            mock_result.error = None
            mock_run_command.return_value = mock_result

            result = tool(operation="validate", command="#check Nat")

            assert result.success == True
            assert result.operation == "validate"
            assert result.result["response"] == "Nat : Type"

    def test_validate_failure(self):
        """Test validation failure."""
        tool = LeanLLMTool()

        with patch.object(tool.prover, 'run_command') as mock_run_command:
            mock_result = Mock()
            mock_result.success = False
            mock_result.response = "[error] unknown identifier"
            mock_result.error = "unknown identifier"
            mock_run_command.return_value = mock_result

            result = tool(operation="validate", command="#check InvalidName")

            assert result.success == False
            assert result.result["error"] == "unknown identifier"

    def test_validate_disabled(self):
        """Test validate operation when disabled."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=False, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)

        result = tool(operation="validate", command="#check Nat")

        assert result.success == False
        assert "not enabled" in result.message


class TestRetrieveOperation:
    """Test the retrieve operation."""

    def test_retrieve_no_session(self):
        """Test retrieve operation with no active session."""
        tool = LeanLLMTool()

        result = tool(operation="retrieve", info_type="panel")

        assert result.success == True
        assert result.operation == "retrieve"
        assert result.result["session_active"] == False
        assert "capabilities" in result.result

    def test_retrieve_with_session(self):
        """Test retrieve operation with active session."""
        tool = LeanLLMTool()

        # Set up mock session
        tool.current_session = {
            "type": "interactive",
            "code": "theorem test : True := by sorry"
        }

        with patch.object(tool.agent, 'get_interactive_panel') as mock_panel:
            mock_panel.return_value = {
                "current_code": "theorem test : True := by sorry",
                "messages": [],
                "goals": [],
                "editable_clauses": {"sorry_0": "sorry: sorry"}
            }

            result = tool(operation="retrieve", info_type="panel")

            assert result.success == True
            assert "current_code" in result.result

    def test_retrieve_current_code(self):
        """Test retrieving current code."""
        tool = LeanLLMTool()
        tool.current_session = {
            "type": "interactive",
            "code": "theorem test : True := trivial"
        }

        result = tool(operation="retrieve", info_type="current_code")

        assert result.success == True
        assert result.result["current_code"] == "theorem test : True := trivial"

    def test_retrieve_suggestions(self):
        """Test retrieving suggestions."""
        tool = LeanLLMTool()
        tool.current_session = {"type": "interactive", "code": "test"}

        with patch.object(tool.agent, 'suggest_next_actions') as mock_suggest:
            mock_suggest.return_value = ["Fix compilation errors", "Work on sorry clauses"]

            result = tool(operation="retrieve", info_type="suggestions")

            assert result.success == True
            assert result.result["suggestions"] == ["Fix compilation errors", "Work on sorry clauses"]

    def test_retrieve_disabled(self):
        """Test retrieve operation when disabled."""
        caps = ToolCapabilities(execute=True, edit_clause=False, add_structure=False, validate=False, retrieve=False)
        tool = LeanLLMTool(capabilities=caps)

        result = tool(operation="retrieve", info_type="panel")

        assert result.success == False
        assert "not enabled" in result.message


class TestSessionManagement:
    """Test session management functionality."""

    def test_reset_session(self):
        """Test session reset."""
        tool = LeanLLMTool()

        # Set up some session data
        tool.current_session = {"type": "interactive", "code": "test"}
        tool.session_history = [{"operation": "test"}]

        tool.reset_session()

        assert tool.current_session is None
        assert len(tool.session_history) == 0

    def test_session_summary(self):
        """Test getting session summary."""
        tool = LeanLLMTool()

        summary = tool.get_session_summary()

        assert "session_active" in summary
        assert "capabilities" in summary
        assert "mathlib_enabled" in summary
        assert summary["session_active"] == False
        assert summary["history_length"] == 0

    def test_session_history_tracking(self):
        """Test that operations are tracked in session history."""
        tool = LeanLLMTool()

        with patch.object(tool.prover, 'run_command') as mock_run:
            mock_result = Mock()
            mock_result.success = True
            mock_result.response = "Success"
            mock_result.error = None
            mock_run.return_value = mock_result

            tool(operation="validate", command="#check Nat")

            assert len(tool.session_history) == 1
            assert tool.session_history[0]["operation"] == "validate"
            assert tool.session_history[0]["command"] == "#check Nat"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_unknown_operation(self):
        """Test handling of unknown operations."""
        tool = LeanLLMTool()

        result = tool(operation="invalid_operation")

        assert result.success == False
        assert ("unknown operation" in result.message.lower() or
                "not enabled" in result.message.lower())
        assert result.error is not None

    def test_exception_handling(self):
        """Test handling of exceptions during operation."""
        tool = LeanLLMTool()

        # Mock an exception
        with patch.object(tool.prover, 'run') as mock_run:
            mock_run.side_effect = Exception("Test exception")

            result = tool(operation="execute", code="test")

            assert result.success == False
            assert "error executing operation" in result.message.lower()
            assert "test exception" in result.error.lower()


class TestConvenienceFunctions:
    """Test convenience functions for tool creation."""

    def test_create_basic_tool(self):
        """Test basic tool creation."""
        tool = create_basic_tool()

        assert tool.capabilities.execute == True
        assert tool.capabilities.edit_clause == False
        assert tool.capabilities.add_structure == False
        assert tool.capabilities.validate == True
        assert tool.capabilities.retrieve == False

    def test_create_interactive_tool(self):
        """Test interactive tool creation."""
        tool = create_interactive_tool()

        assert tool.capabilities.execute == True
        assert tool.capabilities.edit_clause == True
        assert tool.capabilities.add_structure == True
        assert tool.capabilities.validate == True
        assert tool.capabilities.retrieve == True

    def test_create_validation_tool(self):
        """Test validation tool creation."""
        tool = create_validation_tool()

        assert tool.capabilities.execute == False
        assert tool.capabilities.edit_clause == False
        assert tool.capabilities.add_structure == False
        assert tool.capabilities.validate == True
        assert tool.capabilities.retrieve == True

    def test_get_qwen3_tool_config(self):
        """Test Qwen3 configuration generation."""
        tool = LeanLLMTool()
        config = get_qwen3_tool_config(tool)

        assert "name" in config
        assert "description" in config
        assert "parameters" in config
        assert "function" in config
        assert config["name"] == "lean4_tool"
        assert config["function"] == tool


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_interactive_workflow(self):
        """Test a complete interactive theorem development workflow."""
        tool = create_interactive_tool()

        # Mock all necessary methods
        with patch.object(tool.agent, 'load_theorem') as mock_load, \
             patch.object(tool.agent, 'edit_clause') as mock_edit, \
             patch.object(tool.agent, 'get_interactive_panel') as mock_panel:

            # Mock load result
            mock_load.return_value = {"success": True}

            # Mock edit result
            mock_edit.return_value = {
                "edit_successful": True,
                "clause_type": "sorry",
                "updated_code": "theorem test : True := trivial",
                "compilation_result": {"success": True, "has_errors": False, "messages": []}
            }

            # Mock panel result
            mock_panel.return_value = {
                "current_code": "theorem test : True := trivial",
                "messages": ["Compilation successful"],
                "goals": [],
                "editable_clauses": {}
            }

            # Step 1: Load theorem
            result1 = tool(
                operation="edit_clause",
                theorem_code="theorem test : True := by sorry",
                clause_id="sorry_0",
                new_content="trivial"
            )
            assert result1.success == True

            # Step 2: Retrieve current state
            result2 = tool(operation="retrieve", info_type="current_code")
            assert result2.success == True

            # Step 3: Get suggestions
            with patch.object(tool.agent, 'suggest_next_actions') as mock_suggest:
                mock_suggest.return_value = ["Proof looks complete!"]
                result3 = tool(operation="retrieve", info_type="suggestions")
                assert result3.success == True
                assert result3.result["suggestions"] == ["Proof looks complete!"]

    def test_basic_execution_workflow(self):
        """Test basic execution workflow."""
        tool = create_basic_tool()

        with patch.object(tool.prover, 'run') as mock_run, \
             patch.object(tool.prover, 'run_command') as mock_run_command:

            # Mock successful execution
            mock_result = Mock()
            mock_result.success = True
            mock_result.proof_complete = True
            mock_result.has_sorry = False
            mock_result.response = "Success"
            mock_result.error = None
            mock_result.proof_state = None

            mock_run.return_value = mock_result
            mock_run_command.return_value = mock_result

            # Execute theorem
            result1 = tool(operation="execute", code="theorem test : 1 + 1 = 2 := by simp")
            assert result1.success == True

            # Validate with check
            result2 = tool(operation="validate", command="#check test")
            assert result2.success == True

            # Should have history
            assert len(tool.session_history) == 2


# Pytest fixtures for common test setups
@pytest.fixture
def basic_tool():
    """Fixture providing a basic LLM tool."""
    return create_basic_tool()


@pytest.fixture
def interactive_tool():
    """Fixture providing an interactive LLM tool."""
    return create_interactive_tool()


@pytest.fixture
def validation_tool():
    """Fixture providing a validation-only LLM tool."""
    return create_validation_tool()


@pytest.fixture
def custom_capabilities():
    """Fixture providing custom capabilities configuration."""
    return ToolCapabilities(
        execute=True,
        edit_clause=True,
        add_structure=False,
        validate=True,
        retrieve=False
    )


# Parameterized tests for different tool configurations
@pytest.mark.parametrize("tool_factory", [create_basic_tool, create_interactive_tool, create_validation_tool])
def test_tool_creation_parametrized(tool_factory):
    """Parameterized test for different tool creation methods."""
    tool = tool_factory()

    # All tools should have basic properties
    assert isinstance(tool, LeanLLMTool)
    assert hasattr(tool, 'capabilities')
    assert hasattr(tool, 'mathlib_enabled')

    # Schema generation should work for all tools
    schema = tool.get_tool_schema()
    assert isinstance(schema, dict)
    assert "name" in schema
    assert "parameters" in schema


@pytest.mark.parametrize("operation,should_work", [
    ("execute", True),
    ("edit_clause", False),
    ("add_structure", False),
    ("validate", True),
    ("retrieve", False),
])
def test_basic_tool_operations_parametrized(basic_tool, operation, should_work):
    """Parameterized test for basic tool operation enablement."""
    enabled = basic_tool._is_operation_enabled(operation)
    assert enabled == should_work


if __name__ == "__main__":
    # Allow running as script for development/debugging
    print("Running Lean 4 LLM Tool Tests...")
    pytest.main([__file__, "-v"])
