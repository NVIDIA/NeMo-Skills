"""
LLM Tool for Lean 4 Interactive Development

A comprehensive, configurable tool designed for LLM agents (like Qwen3) to interact with Lean 4.
Provides a single tool interface with JSON schema-driven operations that can be dynamically
enabled/disabled based on agent capabilities and requirements.

Features:
- Single tool call interface with multiple operation modes
- Dynamic capability configuration (enable/disable features as needed)
- Comprehensive JSON schema validation
- Support for all lean4 module functionality
- Designed for LLM agent workflows
- Thread-safe for concurrent LLM agent usage

Operations supported:
1. execute: Run Lean code or theorems
2. edit_clause: Edit specific parts of interactive theorems
3. add_structure: Add new proof structure to theorems
4. validate: Run validation commands (#check, #eval, etc.)
5. retrieve: Get current code state and information
"""

import json
import threading
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .prover import LeanProver, ProofResult
from .interactive_agent import InteractiveLeanAgent


class OperationType(Enum):
    """Supported operation types."""
    EXECUTE = "execute"
    EDIT_CLAUSE = "edit_clause"
    ADD_STRUCTURE = "add_structure"
    VALIDATE = "validate"
    RETRIEVE = "retrieve"


@dataclass
class ToolCapabilities:
    """Configuration for which tool capabilities are enabled."""
    execute: bool = True
    edit_clause: bool = True
    add_structure: bool = True
    validate: bool = True
    retrieve: bool = True


@dataclass
class ToolResult:
    """Standardized result format for all tool operations."""
    success: bool
    operation: str
    result: Dict[str, Any]
    message: str
    error: Optional[str] = None


class LeanLLMTool:
    """
    Thread-safe comprehensive LLM tool for Lean 4 interactive development.

    Designed to be used by LLM agents like Qwen3 with a single tool call interface
    that handles multiple operation types through JSON schema validation.

    Each thread (LLM agent) gets its own isolated session state while sharing
    the same tool instance safely.
    """

    def __init__(self, capabilities: ToolCapabilities = None, mathlib_enabled: bool = True):
        """
        Initialize the thread-safe Lean LLM tool.

        Args:
            capabilities: Which tool capabilities to enable (default: all enabled)
            mathlib_enabled: Whether to enable mathlib support
        """
        self.capabilities = capabilities or ToolCapabilities()
        self.mathlib_enabled = mathlib_enabled

        # Thread-local storage for per-agent state
        self._thread_local = threading.local()
        self._lock = threading.RLock()

        # Instance ID for debugging
        self._instance_id = str(uuid.uuid4())[:8]

    def _get_session_state(self):
        """Get or create thread-local session state."""
        if not hasattr(self._thread_local, 'initialized'):
            # Each thread (LLM agent) gets its own state
            self._thread_local.prover = LeanProver(mathlib_enabled=self.mathlib_enabled)
            self._thread_local.agent = InteractiveLeanAgent(mathlib_enabled=self.mathlib_enabled)
            self._thread_local.current_session = None
            self._thread_local.session_history = []
            self._thread_local.thread_id = threading.get_ident()
            self._thread_local.session_id = str(uuid.uuid4())[:8]
            self._thread_local.initialized = True

        return self._thread_local

    @property
    def prover(self) -> LeanProver:
        """Get thread-local prover instance."""
        session = self._get_session_state()
        return session.prover

    @property
    def agent(self) -> InteractiveLeanAgent:
        """Get thread-local agent instance."""
        session = self._get_session_state()
        return session.agent

    @property
    def current_session(self):
        """Get thread-local current session."""
        session = self._get_session_state()
        return session.current_session

    @current_session.setter
    def current_session(self, value):
        """Set thread-local current session."""
        session = self._get_session_state()
        session.current_session = value

    @property
    def session_history(self) -> List[Dict[str, Any]]:
        """Get thread-local session history."""
        session = self._get_session_state()
        return session.session_history

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Generate the JSON schema for the tool based on enabled capabilities.

        Returns:
            JSON schema that defines the tool interface for LLM agents
        """
        # Base schema structure
        schema = {
            "name": "lean4_tool",
            "description": "Interact with Lean 4 theorem prover for mathematical reasoning and proof development.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The type of operation to perform",
                        "enum": []
                    }
                },
                "required": ["operation"]
            }
        }

        # Add operations based on enabled capabilities
        operations = []
        operation_schemas = {}

        if self.capabilities.execute:
            operations.append("execute")
            operation_schemas["execute"] = {
                "description": "Execute Lean code or prove theorems",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Lean code to execute (theorem, definition, etc.)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["proof", "command"],
                        "description": "Execution mode: 'proof' for theorems, 'command' for other Lean commands",
                        "default": "proof"
                    }
                },
                "required": ["code"]
            }

        if self.capabilities.edit_clause:
            operations.append("edit_clause")
            operation_schemas["edit_clause"] = {
                "description": "Edit specific clauses in interactive theorem development",
                "properties": {
                    "theorem_code": {
                        "type": "string",
                        "description": "Full theorem code to load (if starting new session)"
                    },
                    "clause_id": {
                        "type": "string",
                        "description": "ID of the clause to edit (e.g., 'sorry_0', 'have_h1')"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to replace the clause"
                    }
                },
                "required": ["clause_id", "new_content"]
            }

        if self.capabilities.add_structure:
            operations.append("add_structure")
            operation_schemas["add_structure"] = {
                "description": "Add new proof structure to theorem",
                "properties": {
                    "theorem_code": {
                        "type": "string",
                        "description": "Full theorem code to load (if starting new session)"
                    },
                    "structure_lines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lines of proof structure to add"
                    }
                },
                "required": ["structure_lines"]
            }

        if self.capabilities.validate:
            operations.append("validate")
            operation_schemas["validate"] = {
                "description": "Run Lean validation commands for checking and evaluation",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Validation command to run (e.g., '#check Nat.add_comm', '#eval 2 + 2')"
                    }
                },
                "required": ["command"]
            }

        if self.capabilities.retrieve:
            operations.append("retrieve")
            operation_schemas["retrieve"] = {
                "description": "Retrieve current code state and development information",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["current_code", "editable_clauses", "messages", "goals", "suggestions", "panel"],
                        "description": "Type of information to retrieve",
                        "default": "panel"
                    }
                },
                "required": []
            }

        # Update schema with enabled operations
        schema["parameters"]["properties"]["operation"]["enum"] = operations

        # Add conditional schemas based on operation
        if operations:
            schema["parameters"]["allOf"] = []
            for op in operations:
                schema["parameters"]["allOf"].append({
                    "if": {"properties": {"operation": {"const": op}}},
                    "then": {
                        "properties": {**operation_schemas[op]["properties"]},
                        "required": operation_schemas[op]["required"]
                    }
                })

        return schema

    def get_tool_description(self) -> str:
        """Generate a dynamic tool description based on enabled capabilities."""
        capabilities = []
        if self.capabilities.execute:
            capabilities.append("execute Lean code and theorems")
        if self.capabilities.edit_clause:
            capabilities.append("edit theorem clauses interactively")
        if self.capabilities.add_structure:
            capabilities.append("add proof structure")
        if self.capabilities.validate:
            capabilities.append("validate code with #check/#eval")
        if self.capabilities.retrieve:
            capabilities.append("retrieve development state")

        cap_text = ", ".join(capabilities) if capabilities else "no operations"

        return (
            f"Lean 4 theorem prover tool with capabilities: {cap_text}. "
            f"Supports mathlib: {'yes' if self.mathlib_enabled else 'no'}. "
            "Use for mathematical reasoning, theorem proving, and proof development."
        )

    def __call__(self, operation: str, **kwargs) -> ToolResult:
        """
        Main tool interface - handle all operations through single call.
        Thread-safe: each LLM agent gets isolated session state.

        Args:
            operation: The operation type to perform
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with operation results
        """
        try:
            # Validate operation is enabled
            if not self._is_operation_enabled(operation):
                return ToolResult(
                    success=False,
                    operation=operation,
                    result={},
                    message=f"Operation '{operation}' is not enabled",
                    error=f"Operation '{operation}' is disabled in current configuration"
                )

            # Route to appropriate handler
            if operation == OperationType.EXECUTE.value:
                return self._handle_execute(**kwargs)
            elif operation == OperationType.EDIT_CLAUSE.value:
                return self._handle_edit_clause(**kwargs)
            elif operation == OperationType.ADD_STRUCTURE.value:
                return self._handle_add_structure(**kwargs)
            elif operation == OperationType.VALIDATE.value:
                return self._handle_validate(**kwargs)
            elif operation == OperationType.RETRIEVE.value:
                return self._handle_retrieve(**kwargs)
            else:
                return ToolResult(
                    success=False,
                    operation=operation,
                    result={},
                    message=f"Unknown operation: {operation}",
                    error=f"Operation '{operation}' is not recognized"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                operation=operation,
                result={},
                message=f"Error executing operation: {str(e)}",
                error=str(e)
            )

    def _is_operation_enabled(self, operation: str) -> bool:
        """Check if an operation is enabled in current configuration."""
        operation_map = {
            OperationType.EXECUTE.value: self.capabilities.execute,
            OperationType.EDIT_CLAUSE.value: self.capabilities.edit_clause,
            OperationType.ADD_STRUCTURE.value: self.capabilities.add_structure,
            OperationType.VALIDATE.value: self.capabilities.validate,
            OperationType.RETRIEVE.value: self.capabilities.retrieve,
        }
        return operation_map.get(operation, False)

    def _handle_execute(self, code: str, mode: str = "proof") -> ToolResult:
        """Handle code execution operations."""
        if mode == "command":
            result = self.prover.run_command(code)
        else:
            result = self.prover.run(code)

        # Store in thread-local session history
        self.session_history.append({
            "operation": "execute",
            "code": code,
            "mode": mode,
            "result": self._safe_result_to_dict(result),
            "timestamp": self._get_timestamp()
        })

        return ToolResult(
            success=result.success,
            operation="execute",
            result={
                "success": result.success,
                "proof_complete": result.proof_complete,
                "has_sorry": result.has_sorry,
                "response": result.response,
                "error": result.error,
                "proof_state": result.proof_state
            },
            message=f"Execution {'successful' if result.success else 'failed'}: {result.response[:100]}..."
        )

    def _handle_edit_clause(self, clause_id: str, new_content: str, theorem_code: str = None) -> ToolResult:
        """Handle interactive clause editing."""
        # Load theorem if provided
        if theorem_code:
            load_result = self.agent.load_theorem(theorem_code)
            self.current_session = {
                "type": "interactive",
                "code": theorem_code,
                "load_result": load_result
            }

        if not self.current_session or self.current_session.get("type") != "interactive":
            return ToolResult(
                success=False,
                operation="edit_clause",
                result={},
                message="No interactive session active. Provide theorem_code to start.",
                error="Interactive session required for clause editing"
            )

        # Edit the clause
        edit_result = self.agent.edit_clause(clause_id, new_content)

        # Update session
        if edit_result.get("edit_successful"):
            self.current_session["code"] = edit_result["updated_code"]

        # Store in thread-local history
        self.session_history.append({
            "operation": "edit_clause",
            "clause_id": clause_id,
            "new_content": new_content,
            "result": edit_result,
            "timestamp": self._get_timestamp()
        })

        compilation_result = edit_result.get("compilation_result", {})

        return ToolResult(
            success=edit_result.get("edit_successful", False),
            operation="edit_clause",
            result={
                "edit_successful": edit_result.get("edit_successful", False),
                "clause_type": edit_result.get("clause_type"),
                "updated_code": edit_result.get("updated_code"),
                "compilation_success": compilation_result.get("success", False),
                "has_errors": compilation_result.get("has_errors", False),
                "messages": compilation_result.get("messages", []),
                "editable_clauses": compilation_result.get("editable_clauses", [])
            },
            message=f"Clause edit {'successful' if edit_result.get('edit_successful') else 'failed'}"
        )

    def _handle_add_structure(self, structure_lines: List[str], theorem_code: str = None) -> ToolResult:
        """Handle adding new proof structure."""
        # Load theorem if provided
        if theorem_code:
            load_result = self.agent.load_theorem(theorem_code)
            self.current_session = {
                "type": "interactive",
                "code": theorem_code,
                "load_result": load_result
            }

        if not self.current_session or self.current_session.get("type") != "interactive":
            return ToolResult(
                success=False,
                operation="add_structure",
                result={},
                message="No interactive session active. Provide theorem_code to start.",
                error="Interactive session required for adding structure"
            )

        # Add the structure
        structure_result = self.agent.add_proof_structure(structure_lines)

        # Update session
        if structure_result.get("edit_successful"):
            panel = self.agent.get_interactive_panel()
            self.current_session["code"] = panel["current_code"]

        # Store in thread-local history
        self.session_history.append({
            "operation": "add_structure",
            "structure_lines": structure_lines,
            "result": structure_result,
            "timestamp": self._get_timestamp()
        })

        compilation_result = structure_result.get("compilation_result", {})

        return ToolResult(
            success=structure_result.get("edit_successful", False),
            operation="add_structure",
            result={
                "edit_successful": structure_result.get("edit_successful", False),
                "updated_code": structure_result.get("updated_code"),
                "compilation_success": compilation_result.get("success", False),
                "has_errors": compilation_result.get("has_errors", False),
                "messages": compilation_result.get("messages", []),
                "editable_clauses": compilation_result.get("editable_clauses", [])
            },
            message=f"Structure addition {'successful' if structure_result.get('edit_successful') else 'failed'}"
        )

    def _handle_validate(self, command: str) -> ToolResult:
        """Handle validation commands."""
        # Validation commands should work with the prover directly
        result = self.prover.run_command(command)

        # Store in thread-local history
        self.session_history.append({
            "operation": "validate",
            "command": command,
            "result": self._safe_result_to_dict(result),
            "timestamp": self._get_timestamp()
        })

        return ToolResult(
            success=result.success,
            operation="validate",
            result={
                "success": result.success,
                "response": result.response,
                "error": result.error
            },
            message=f"Validation {'successful' if result.success else 'failed'}: {command}"
        )

    def _handle_retrieve(self, info_type: str = "panel") -> ToolResult:
        """Handle information retrieval operations."""
        if not self.current_session or self.current_session.get("type") != "interactive":
            # If no interactive session, provide basic info
            result_data = {
                "session_active": False,
                "session_history_length": len(self.session_history),
                "capabilities": asdict(self.capabilities),
                "thread_info": {
                    "thread_id": getattr(self._get_session_state(), 'thread_id', None),
                    "session_id": getattr(self._get_session_state(), 'session_id', None),
                    "instance_id": self._instance_id
                }
            }

            return ToolResult(
                success=True,
                operation="retrieve",
                result=result_data,
                message="No active interactive session. Basic info provided."
            )

        # Get information based on type
        if info_type == "current_code":
            result_data = {"current_code": self.current_session.get("code", "")}

        elif info_type == "editable_clauses":
            clauses = getattr(self.agent, 'editable_clauses', {})
            result_data = {
                "editable_clauses": {
                    cid: {"type": clause.clause_type, "content": clause.content}
                    for cid, clause in clauses.items()
                }
            }

        elif info_type == "messages":
            result_data = {"messages": [str(msg) for msg in self.agent.current_messages]}

        elif info_type == "goals":
            result_data = {"goals": [str(goal) for goal in self.agent.current_goals]}

        elif info_type == "suggestions":
            suggestions = self.agent.suggest_next_actions()
            result_data = {"suggestions": suggestions}

        else:  # info_type == "panel" (default)
            panel = self.agent.get_interactive_panel()
            result_data = panel

        return ToolResult(
            success=True,
            operation="retrieve",
            result=result_data,
            message=f"Retrieved {info_type} information"
        )

    def _safe_result_to_dict(self, result) -> Dict[str, Any]:
        """Safely convert a result object to a dictionary."""
        try:
            if hasattr(result, '__dict__'):
                return {
                    key: str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
                    for key, value in result.__dict__.items()
                }
            else:
                return {"result": str(result)}
        except:
            return {"result": "Failed to serialize result"}

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()

    def reset_session(self):
        """Reset the current thread's session and clear history."""
        session_state = self._get_session_state()
        session_state.current_session = None
        session_state.session_history = []

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current thread's session state."""
        session_state = self._get_session_state()
        return {
            "capabilities": asdict(self.capabilities),
            "mathlib_enabled": self.mathlib_enabled,
            "session_active": self.current_session is not None,
            "session_type": self.current_session.get("type") if self.current_session else None,
            "history_length": len(self.session_history),
            "thread_info": {
                "thread_id": getattr(session_state, 'thread_id', None),
                "session_id": getattr(session_state, 'session_id', None),
                "instance_id": self._instance_id
            }
        }

    def get_thread_info(self) -> Dict[str, Any]:
        """Get debugging info about current thread's state."""
        session_state = self._get_session_state()
        return {
            "instance_id": self._instance_id,
            "thread_id": getattr(session_state, 'thread_id', None),
            "session_id": getattr(session_state, 'session_id', None),
            "prover_thread_info": self.prover.get_thread_info(),
            "agent_thread_info": self.agent.get_thread_info(),
            "session_active": self.current_session is not None,
            "history_length": len(self.session_history)
        }


# Convenience functions for easy tool creation with different configurations

def create_basic_tool(mathlib_enabled: bool = True) -> LeanLLMTool:
    """Create a basic tool with only execution capabilities."""
    capabilities = ToolCapabilities(
        execute=True,
        edit_clause=False,
        add_structure=False,
        validate=True,
        retrieve=False
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


def create_interactive_tool(mathlib_enabled: bool = True) -> LeanLLMTool:
    """Create an interactive tool with editing capabilities."""
    capabilities = ToolCapabilities(
        execute=True,
        edit_clause=True,
        add_structure=True,
        validate=True,
        retrieve=True
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


def create_validation_tool(mathlib_enabled: bool = True) -> LeanLLMTool:
    """Create a tool focused on validation and checking."""
    capabilities = ToolCapabilities(
        execute=False,
        edit_clause=False,
        add_structure=False,
        validate=True,
        retrieve=True
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


# Example usage for LLM integration
def get_qwen3_tool_config(tool: LeanLLMTool) -> Dict[str, Any]:
    """
    Get tool configuration in format suitable for Qwen3 integration.

    Returns:
        Dictionary with tool name, description, and schema
    """
    schema = tool.get_tool_schema()
    return {
        "name": schema["name"],
        "description": tool.get_tool_description(),
        "parameters": schema["parameters"],
        "function": tool  # The callable tool instance
    }
