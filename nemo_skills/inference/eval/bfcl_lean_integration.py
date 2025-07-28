#!/usr/bin/env python3
"""
BFCL Integration for Lean 4 Tools

This module provides BFCL-compatible wrappers for Lean 4 tools that preserve
thread-local session state during BFCL's multi-turn evaluation process.
"""

import threading
import logging
from typing import Dict, Any, Optional

from nemo_skills.code_execution.lean4 import create_interactive_tool, LeanLLMTool
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Global registry to map thread IDs to LeanLLMTool instances
_THREAD_TOOL_REGISTRY: Dict[int, LeanLLMTool] = {}
_REGISTRY_LOCK = threading.RLock()


class LeanAPI:
    """BFCL-compatible wrapper for Lean 4 tools that preserves thread-local state."""

    def __init__(self):
        """Initialize the BFCL-compatible Lean API wrapper."""
        self._thread_id = threading.get_ident()
        LOG.debug(f"LeanAPI initialized for thread {self._thread_id}")

    def _get_tool_instance(self) -> LeanLLMTool:
        """Get the LeanLLMTool instance for the current thread."""
        thread_id = threading.get_ident()

        with _REGISTRY_LOCK:
            if thread_id not in _THREAD_TOOL_REGISTRY:
                tool = create_interactive_tool(mathlib_enabled=True)
                _THREAD_TOOL_REGISTRY[thread_id] = tool
                LOG.info(f"Created new LeanLLMTool for thread {thread_id}")
            else:
                LOG.debug(f"Using existing LeanLLMTool for thread {thread_id}")

            return _THREAD_TOOL_REGISTRY[thread_id]

    def _load_scenario(self, config: Dict[str, Any], long_context: bool = False):
        """BFCL compatibility method - required for non-stateless classes."""
        LOG.debug(f"Loading scenario for thread {threading.get_ident()}: {config}")
        pass

    def execute_lean_code(self, code: str, mode: str = "proof") -> str:
        """Execute Lean 4 code such as theorems, definitions, or commands."""
        tool = self._get_tool_instance()
        result = tool.execute_lean_code(code=code, mode=mode)

        if result.success:
            return result.result.get("response", "Execution completed successfully")
        else:
            return f"Error: {result.result.get('error', result.message)}"

    def start_interactive_theorem(self, theorem_code: str) -> str:
        """Load a theorem for interactive development."""
        tool = self._get_tool_instance()
        result = tool.start_interactive_theorem(theorem_code=theorem_code)

        if result.success:
            editable_clauses = result.result.get("editable_clauses", [])
            proof_complete = result.result.get("proof_complete", False)
            has_errors = result.result.get("has_errors", False)

            if has_errors:
                messages = result.result.get("messages", [])
                error_msgs = [str(msg) for msg in messages if getattr(msg, 'severity', 'error') == 'error']
                return f"Interactive theorem loaded with errors: {'; '.join(error_msgs[:2])}"
            elif proof_complete:
                return f"Interactive theorem loaded and proof is already complete! Editable clauses: {editable_clauses}"
            else:
                return f"Interactive theorem loaded with {len(editable_clauses)} editable clauses: {editable_clauses}"
        else:
            return f"Error loading theorem: {result.error}"

    def edit_proof_clause(self, clause_id: str, new_content: str) -> str:
        """Edit a specific clause or part of an interactive theorem proof."""
        tool = self._get_tool_instance()
        result = tool.edit_proof_clause(clause_id=clause_id, new_content=new_content)

        if result.success:
            compilation_result = result.result.get("compilation_result", {})
            has_errors = compilation_result.get("has_errors", False)
            proof_complete = compilation_result.get("proof_complete", False)
            messages = compilation_result.get("messages", [])

            if has_errors:
                error_msgs = [str(msg) for msg in messages if getattr(msg, 'severity', 'error') == 'error']
                return f"Clause '{clause_id}' edited but has compilation errors: {'; '.join(error_msgs[:2])}"
            elif proof_complete:
                return f"Clause '{clause_id}' edited successfully. Proof is now complete!"
            else:
                warning_msgs = [str(msg) for msg in messages if getattr(msg, 'severity', 'warning') == 'warning']
                if warning_msgs:
                    return f"Clause '{clause_id}' edited successfully (warnings: {'; '.join(warning_msgs[:1])})"
                else:
                    return f"Clause '{clause_id}' edited successfully. Proof still incomplete."
        else:
            return f"Error editing clause: {result.error}"

    def add_proof_structure(self, structure_lines: list) -> str:
        """Add new proof structure lines to an interactive theorem."""
        tool = self._get_tool_instance()
        result = tool.add_proof_structure(structure_lines=structure_lines)

        if result.success:
            compilation_result = result.result.get("compilation_result", {})
            has_errors = compilation_result.get("has_errors", False)
            proof_complete = compilation_result.get("proof_complete", False)
            messages = compilation_result.get("messages", [])

            if has_errors:
                error_msgs = [str(msg) for msg in messages if getattr(msg, 'severity', 'error') == 'error']
                return f"Added {len(structure_lines)} structure lines but introduced errors: {'; '.join(error_msgs[:2])}"
            elif proof_complete:
                return f"Added {len(structure_lines)} structure lines. Proof is now complete!"
            else:
                return f"Added {len(structure_lines)} structure lines to proof (still incomplete)"
        else:
            return f"Error adding structure: {result.error}"

    def validate_lean(self, command: str) -> str:
        """Run Lean validation commands like #check, #eval, or #print."""
        tool = self._get_tool_instance()
        result = tool.validate_lean(command=command)

        if result.success:
            return result.result.get("response", "Validation completed")
        else:
            return f"Validation error: {result.result.get('error', result.message)}"

    def get_proof_state(self, info_type: str = "full_panel") -> str:
        """Get the current state of an interactive proof session."""
        tool = self._get_tool_instance()
        result = tool.get_proof_state(info_type=info_type)

        if result.success:
            state = result.result
            if info_type == "current_code":
                return state.get("current_code", "No code available")
            elif info_type == "editable_clauses":
                clauses = state.get("editable_clauses", [])
                return f"Editable clauses: {clauses}"
            else:
                current_code = state.get("current_code", "")
                return f"Current code:\n{current_code}"
        else:
            return f"Error getting proof state: {result.error}"


def register_lean_tools():
    """Register Lean tools with BFCL's tool registry."""
    try:
        from nemo_skills.inference.eval.bfcl_registry import register_tool_class
        register_tool_class("LeanAPI", "nemo_skills.inference.eval.bfcl_lean_integration", stateless=False)
        LOG.info("Successfully registered LeanAPI with BFCL")
    except ImportError:
        LOG.warning("BFCL registry not available - tools may not work in BFCL evaluation")


def get_thread_tool_info() -> Dict[str, Any]:
    """Get debug information about thread-tool mapping."""
    with _REGISTRY_LOCK:
        return {
            "active_threads": list(_THREAD_TOOL_REGISTRY.keys()),
            "current_thread": threading.get_ident(),
            "total_tool_instances": len(_THREAD_TOOL_REGISTRY)
        }


# Auto-register when imported
register_lean_tools()
