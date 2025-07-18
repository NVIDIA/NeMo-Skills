"""
Simple Lean 4 Prover using LeanInteract with mathlib support.

Focuses on:
- Simple interface for executing proof steps
- Clear distinction between proofs with sorry vs complete proofs
- User-managed proof state with backtracking support
- Mathlib and aesop imports via TempRequireProject
"""

from typing import Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from lean_interact import AutoLeanServer, LeanREPLConfig, TempRequireProject, Command, ProofStep

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    """Result of executing a proof step."""
    success: bool
    goals: List[str]
    proof_complete: bool
    has_sorry: bool
    response_text: str
    error_message: Optional[str] = None


class LeanProver:
    """Simple Lean 4 prover with mathlib support."""

    def __init__(self, use_mathlib: bool = True):
        """Initialize the prover with optional mathlib support."""
        self.use_mathlib = use_mathlib
        self.server = None
        self.project = None
        self._initialize_server()

    def _initialize_server(self):
        """Initialize the Lean server with mathlib if requested."""
        if self.use_mathlib:
            self.project = TempRequireProject(require="mathlib")
            config = LeanREPLConfig(project=self.project)
            self.server = AutoLeanServer(config)
        else:
            config = LeanREPLConfig()
            self.server = AutoLeanServer(config)

        # Add standard imports
        if self.use_mathlib:
            imports = [
                "import Mathlib",
                "import Aesop",
                "import Mathlib.Data.Real.Basic",
                "import Mathlib.Tactic",
                "import Mathlib.Data.Nat.Basic"
            ]
            for imp in imports:
                self.server.run(Command(cmd=imp))

    def run_proof_step(self, theorem_statement: str, tactic: str) -> ProofResult:
        """
        Execute a single proof step.

        Args:
            theorem_statement: The theorem to prove (e.g., "theorem test : 1 + 1 = 2")
            tactic: The tactic to apply (e.g., "norm_num")

        Returns:
            ProofResult with success status, goals, and completion info
        """
        try:
            # Format as a complete theorem with proof
            full_statement = f"{theorem_statement} := by {tactic}"

            # Execute the command
            response = self.server.run(Command(cmd=full_statement))

            # Parse the response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error executing proof step: {e}")
            return ProofResult(
                success=False,
                goals=[],
                proof_complete=False,
                has_sorry=False,
                response_text=str(e),
                error_message=str(e)
            )

    def run_tactic_sequence(self, theorem_statement: str, tactics: List[str]) -> ProofResult:
        """
        Execute a sequence of tactics.

        Args:
            theorem_statement: The theorem to prove
            tactics: List of tactics to apply in order

        Returns:
            ProofResult for the final state
        """
        try:
            # Format tactics with proper indentation
            tactic_sequence = "\n  ".join(tactics)
            full_statement = f"{theorem_statement} := by\n  {tactic_sequence}"

            response = self.server.run(Command(cmd=full_statement))
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error executing tactic sequence: {e}")
            return ProofResult(
                success=False,
                goals=[],
                proof_complete=False,
                has_sorry=False,
                response_text=str(e),
                error_message=str(e)
            )

    def check_proof_with_sorry(self, theorem_statement: str, proof_outline: str) -> ProofResult:
        """
        Check a proof that may contain 'sorry' placeholders.

        Args:
            theorem_statement: The theorem to prove
            proof_outline: Proof text that may contain sorry

        Returns:
            ProofResult indicating if proof structure is valid (even with sorry)
        """
        try:
            # Format as complete statement
            if proof_outline.startswith("by"):
                full_statement = f"{theorem_statement} := {proof_outline}"
            else:
                full_statement = f"{theorem_statement} := by {proof_outline}"

            response = self.server.run(Command(cmd=full_statement))
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error checking proof with sorry: {e}")
            return ProofResult(
                success=False,
                goals=[],
                proof_complete=False,
                has_sorry=True,
                response_text=str(e),
                error_message=str(e)
            )

    def _parse_response(self, response) -> ProofResult:
        """Parse the Lean server response into a ProofResult."""
        if not response:
            return ProofResult(
                success=False,
                goals=[],
                proof_complete=False,
                has_sorry=False,
                response_text="No response from server",
                error_message="No response from server"
            )

        # Empty message list means success (no errors)
        if not response.messages:
            return ProofResult(
                success=True,
                goals=[],
                proof_complete=True,
                has_sorry=False,
                response_text="",
                error_message=None
            )

        response_text = ""
        has_errors = False
        has_sorry = False
        goals = []

        for msg in response.messages:
            response_text += f"{msg.severity}: {msg.data}\n"

            # Check for sorry
            if "sorry" in msg.data.lower() or "declaration uses 'sorry'" in msg.data:
                has_sorry = True

            # Handle different types of messages
            if msg.severity == "error":
                # "unsolved goals" is not a real error - it's partial success
                if "unsolved goals" in msg.data:
                    # Extract goals from the message
                    goal_matches = re.findall(r'⊢\s*(.+)', msg.data)
                    goals.extend([goal.strip() for goal in goal_matches])
                else:
                    # This is a real error
                    has_errors = True

            # Also check for goals in info messages
            elif msg.severity == "info" and "goals" in msg.data:
                goal_matches = re.findall(r'⊢\s*(.+)', msg.data)
                goals.extend([goal.strip() for goal in goal_matches])

        # Determine success and completion
        success = not has_errors
        proof_complete = success and not goals and not has_sorry

        return ProofResult(
            success=success,
            goals=goals,
            proof_complete=proof_complete,
            has_sorry=has_sorry,
            response_text=response_text.strip(),
            error_message=None if success else response_text.strip()
        )

    def close(self):
        """Clean up resources."""
        if self.server:
            self.server.kill()
        if self.project:
            # TempRequireProject doesn't have a close method, cleanup is automatic
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
def quick_prove(theorem_statement: str, tactic: str, use_mathlib: bool = True) -> ProofResult:
    """Quick proof attempt with a single tactic."""
    with LeanProver(use_mathlib=use_mathlib) as prover:
        return prover.run_proof_step(theorem_statement, tactic)


def quick_prove_sequence(theorem_statement: str, tactics: List[str], use_mathlib: bool = True) -> ProofResult:
    """Quick proof attempt with a sequence of tactics."""
    with LeanProver(use_mathlib=use_mathlib) as prover:
        return prover.run_tactic_sequence(theorem_statement, tactics)
