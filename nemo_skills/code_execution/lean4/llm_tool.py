#!/usr/bin/env python3
"""
LLM Tool for Lean 4 Interactive Development

A comprehensive, configurable tool designed for LLM agents to interact with Lean 4.
Provides BFCL-compatible function interfaces for theorem proving and mathematical reasoning.

Features:
- Direct BFCL function methods (no operation dispatching)
- Dynamic capability configuration (enable/disable features as needed)
- Support for all lean4 module functionality
- Designed for LLM agent workflows
- Thread-safe for concurrent LLM agent usage

BFCL Functions supported:
1. execute_lean_code: Run Lean code or theorems
2. start_interactive_theorem: Load theorem for interactive development
3. edit_proof_clause: Edit specific parts of interactive theorems
4. add_proof_structure: Add new proof structure to theorems
5. validate_lean: Run validation commands (#check, #eval, etc.)
6. get_proof_state: Get current code state and information
"""

import json
import threading
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, asdict

from .prover import LeanProver, ProofResult
from .interactive_agent import InteractiveLeanAgent


@dataclass
class ToolCapabilities:
    """Configure which tool capabilities are enabled."""
    execute_lean_code: bool = True
    start_interactive_theorem: bool = True
    edit_proof_clause: bool = True
    add_proof_structure: bool = True
    validate_lean: bool = True
    get_proof_state: bool = True


@dataclass
class ToolResult:
    """Standardized result format for all tool operations."""
    success: bool
    function_name: str
    result: Dict[str, Any]
    message: str
    error: Optional[str] = None


class LeanLLMTool:
    """
    Thread-safe comprehensive LLM tool for Lean 4 interactive development.

    Designed to be used by LLM agents with direct BFCL function calls
    that handle theorem proving and mathematical reasoning.

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

    # BFCL Function Methods

    def execute_lean_code(self, code: str, mode: str = "proof") -> ToolResult:
        """Execute Lean 4 code such as theorems, definitions, or commands."""
        try:
            if mode == "command":
                result = self.prover.run_command(code)
            else:
                result = self.prover.run(code)

            # Store in thread-local session history
            self.session_history.append({
                "function": "execute_lean_code",
                "code": code,
                "mode": mode,
                "result": self._safe_result_to_dict(result),
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=result.success,
                function_name="execute_lean_code",
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

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="execute_lean_code",
                result={},
                message=f"Error executing code: {str(e)}",
                error=str(e)
            )

    def start_interactive_theorem(self, theorem_code: str) -> ToolResult:
        """Load a theorem for interactive development."""
        try:
            result = self.agent.load_theorem(theorem_code)
            self.current_session = "interactive"

            self.session_history.append({
                "function": "start_interactive_theorem",
                "theorem_code": theorem_code,
                "result": result,
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=result.get("success", True),
                function_name="start_interactive_theorem",
                result=result,
                message=f"Interactive theorem loaded with {len(result.get('editable_clauses', {}))} editable clauses"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="start_interactive_theorem",
                result={},
                message=f"Error loading interactive theorem: {str(e)}",
                error=str(e)
            )

    def edit_proof_clause(self, clause_id: str, new_content: str) -> ToolResult:
        """Edit a specific clause or part of an interactive theorem proof."""
        try:
            if self.current_session != "interactive":
                return ToolResult(
                    success=False,
                    function_name="edit_proof_clause",
                    result={},
                    message="No interactive session active. Use start_interactive_theorem first.",
                    error="No interactive session"
                )

            result = self.agent.edit_clause(clause_id, new_content)

            self.session_history.append({
                "function": "edit_proof_clause",
                "clause_id": clause_id,
                "new_content": new_content,
                "result": result,
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=result.get("success", True),
                function_name="edit_proof_clause",
                result=result,
                message=f"Clause '{clause_id}' edited successfully" if result.get("success") else f"Failed to edit clause '{clause_id}'"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="edit_proof_clause",
                result={},
                message=f"Error editing clause: {str(e)}",
                error=str(e)
            )

    def add_proof_structure(self, structure_lines: List[str]) -> ToolResult:
        """Add new proof structure lines to an interactive theorem."""
        try:
            if self.current_session != "interactive":
                return ToolResult(
                    success=False,
                    function_name="add_proof_structure",
                    result={},
                    message="No interactive session active. Use start_interactive_theorem first.",
                    error="No interactive session"
                )

            result = self.agent.add_proof_structure(structure_lines)

            self.session_history.append({
                "function": "add_proof_structure",
                "structure_lines": structure_lines,
                "result": result,
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=result.get("success", True),
                function_name="add_proof_structure",
                result=result,
                message=f"Added {len(structure_lines)} structure lines to proof"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="add_proof_structure",
                result={},
                message=f"Error adding proof structure: {str(e)}",
                error=str(e)
            )

    def validate_lean(self, command: str) -> ToolResult:
        """Run Lean validation commands like #check, #eval, or #print."""
        try:
            result = self.prover.run_command(command)

            self.session_history.append({
                "function": "validate_lean",
                "command": command,
                "result": self._safe_result_to_dict(result),
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=result.success,
                function_name="validate_lean",
                result={
                    "success": result.success,
                    "response": result.response,
                    "error": result.error
                },
                message=f"Validation {'successful' if result.success else 'failed'}: {result.response[:100]}..."
            )

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="validate_lean",
                result={},
                message=f"Error running validation: {str(e)}",
                error=str(e)
            )

    def get_proof_state(self, info_type: str = "full_panel") -> ToolResult:
        """Get the current state of an interactive proof session."""
        try:
            if self.current_session != "interactive":
                return ToolResult(
                    success=False,
                    function_name="get_proof_state",
                    result={},
                    message="No interactive session active. Use start_interactive_theorem first.",
                    error="No interactive session"
                )

            if info_type == "full_panel":
                result = self.agent.get_interactive_panel()
            elif info_type == "current_code":
                result = {"current_code": self.agent.current_code}
            elif info_type == "messages":
                result = {"messages": self.agent.current_messages}
            elif info_type == "editable_clauses":
                result = {"editable_clauses": self.agent.editable_clauses}
            elif info_type == "suggestions":
                result = self.agent.suggest_next_actions()
            else:
                result = self.agent.get_interactive_panel()

            self.session_history.append({
                "function": "get_proof_state",
                "info_type": info_type,
                "result": result,
                "timestamp": self._get_timestamp()
            })

            return ToolResult(
                success=True,
                function_name="get_proof_state",
                result=result,
                message=f"Retrieved {info_type} information"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                function_name="get_proof_state",
                result={},
                message=f"Error getting proof state: {str(e)}",
                error=str(e)
            )

    def get_bfcl_functions(self) -> List[Dict[str, Any]]:
        """
        Generate BFCL-compatible function definitions.

        Returns:
            List of function definitions compatible with BFCL evaluation framework
        """
        functions = []

        if self.capabilities.execute_lean_code:
            functions.append({
                "name": "execute_lean_code",
                "description": "Execute Lean 4 code such as theorems, definitions, or commands. This is the main function for proving theorems and running Lean code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Lean 4 code to execute. This can be a theorem to prove, a definition, or any other Lean command."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["proof", "command"],
                            "description": "Execution mode: 'proof' for theorems and proofs, 'command' for other Lean commands like #check or #eval",
                            "default": "proof"
                        }
                    },
                    "required": ["code"]
                }
            })

        if self.capabilities.start_interactive_theorem:
            functions.append({
                "name": "start_interactive_theorem",
                "description": "Load a theorem for interactive development. This allows you to work on a theorem step by step, editing specific parts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "theorem_code": {
                            "type": "string",
                            "description": "The complete theorem code to load for interactive editing. Should include theorem declaration and initial proof structure."
                        }
                    },
                    "required": ["theorem_code"]
                }
            })

        if self.capabilities.edit_proof_clause:
            functions.append({
                "name": "edit_proof_clause",
                "description": "Edit a specific clause or part of an interactive theorem proof. Use this to refine specific parts of a proof.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clause_id": {
                            "type": "string",
                            "description": "ID of the clause to edit (e.g., 'sorry_0', 'have_h1', 'main_proof_0')"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content to replace the clause with"
                        }
                    },
                    "required": ["clause_id", "new_content"]
                }
            })

        if self.capabilities.add_proof_structure:
            functions.append({
                "name": "add_proof_structure",
                "description": "Add new proof structure lines to an interactive theorem. Use this to add 'have' statements or other proof structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of proof structure lines to add (e.g., ['have h1 : P := by sorry', 'exact h1'])"
                        }
                    },
                    "required": ["structure_lines"]
                }
            })

        if self.capabilities.validate_lean:
            functions.append({
                "name": "validate_lean",
                "description": "Run Lean validation commands like #check, #eval, or #print to verify types, evaluate expressions, or inspect definitions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Validation command to run (e.g., '#check Nat.add_comm', '#eval 2 + 2', '#print List')"
                        }
                    },
                    "required": ["command"]
                }
            })

        if self.capabilities.get_proof_state:
            functions.append({
                "name": "get_proof_state",
                "description": "Get the current state of an interactive proof session including code, messages, goals, and available clauses to edit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "enum": ["full_panel", "current_code", "messages", "editable_clauses", "suggestions"],
                            "description": "Type of information to retrieve about the current proof state",
                            "default": "full_panel"
                        }
                    },
                    "required": []
                }
            })

        return functions

    def _safe_result_to_dict(self, result: ProofResult) -> Dict[str, Any]:
        """Convert ProofResult to dictionary safely."""
        try:
            return asdict(result)
        except Exception:
            return {
                "success": result.success,
                "proof_complete": result.proof_complete,
                "has_sorry": result.has_sorry,
                "response": result.response,
                "error": result.error
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
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
        execute_lean_code=True,
        start_interactive_theorem=False,
        edit_proof_clause=False,
        add_proof_structure=False,
        validate_lean=True,
        get_proof_state=False
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


def create_interactive_tool(mathlib_enabled: bool = True) -> LeanLLMTool:
    """Create an interactive tool with editing capabilities."""
    capabilities = ToolCapabilities(
        execute_lean_code=True,
        start_interactive_theorem=True,
        edit_proof_clause=True,
        add_proof_structure=True,
        validate_lean=True,
        get_proof_state=True
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


def create_validation_tool(mathlib_enabled: bool = True) -> LeanLLMTool:
    """Create a tool focused on validation and checking."""
    capabilities = ToolCapabilities(
        execute_lean_code=False,
        start_interactive_theorem=False,
        edit_proof_clause=False,
        add_proof_structure=False,
        validate_lean=True,
        get_proof_state=True
    )
    return LeanLLMTool(capabilities=capabilities, mathlib_enabled=mathlib_enabled)


# Example usage for LLM integration
def get_qwen3_tool_config(tool: LeanLLMTool) -> Dict[str, Any]:
    """
    Get tool configuration in format suitable for general LLM integration.

    Returns:
        Dictionary with BFCL functions and tool instance
    """
    return {
        "functions": tool.get_bfcl_functions(),
        "tool_instance": tool,
        "description": "Lean 4 theorem prover with BFCL-compatible functions"
    }
