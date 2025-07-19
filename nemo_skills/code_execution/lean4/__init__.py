"""
Lean 4 Theorem Proving Utilities for NeMo-Skills

This submodule provides comprehensive Lean 4 theorem proving capabilities:

1. LeanProver: Core prover interface with mathlib support
   - Simple interface for executing proof steps
   - Clear distinction between proofs with sorry vs complete proofs
   - User-managed proof state with backtracking support
   - Incremental proof building with tactic manipulation

2. InteractiveLeanAgent: VS Code-like interactive development experience
   - Real-time compiler feedback and messages
   - Position-aware editing with goal state tracking
   - Targeted updates with immediate validation
   - Interactive development workflow for LLM agents

Usage:
    from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent

    # Basic prover usage
    prover = LeanProver(mathlib_enabled=True)
    result = prover.run("theorem test : 1 + 1 = 2 := by simp")

    # Interactive agent usage (mimics VS Code Lean 4 extension)
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    result = agent.load_theorem("theorem demo : True := by sorry")
    agent.edit_clause("sorry_0", "trivial")
"""

from .prover import (
    LeanProver,
    ProofResult,
    ProofInProgress,
)

from .interactive_agent import (
    InteractiveLeanAgent,
    Position,
    LeanMessage,
    ProofGoal,
    EditableClause,
    demo_interactive_agent,
)

__all__ = [
    # Core prover
    'LeanProver',
    'ProofResult',
    'ProofInProgress',

    # Interactive agent
    'InteractiveLeanAgent',
    'Position',
    'LeanMessage',
    'ProofGoal',
    'EditableClause',
    'demo_interactive_agent',
]

# Version info
__version__ = "1.0.0"
__author__ = "NeMo-Skills Team"
__description__ = "Lean 4 theorem proving utilities with interactive development support"
