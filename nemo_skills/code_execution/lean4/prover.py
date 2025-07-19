"""
Simple Lean 4 Prover using LeanInteract with mathlib support.

Focuses on:
- Simple interface for executing proof steps
- Clear distinction between proofs with sorry vs complete proofs
- User-managed proof state with backtracking support
- Mathlib and aesop imports via TempRequireProject
- ProofStep execution for granular tactic application
- Command execution for standalone operations
- Incremental proof building with tactic manipulation
"""

from lean_interact import AutoLeanServer, LeanREPLConfig, Command, ProofStep, TempRequireProject
from lean_interact.interface import CommandResponse, ProofStepResponse, LeanError
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    """Result of a proof execution."""
    success: bool
    proof_complete: bool
    has_sorry: bool
    response: str
    error: Optional[str] = None
    proof_state: Optional[int] = None
    goals: Optional[List[str]] = None

@dataclass
class ProofInProgress:
    """Represents a proof that is being built incrementally."""
    name: str
    statement: str
    initial_proof_state: int
    current_proof_state: int
    tactic_history: List[Tuple[int, str]]  # (from_state, tactic)
    goals: List[str]
    completed: bool = False

class LeanProver:
    """A simple but minimally sufficient Lean 4 prover interface."""

    def __init__(self, mathlib_enabled: bool = True):
        """Initialize the Lean prover with mathlib support."""
        self.mathlib_enabled = mathlib_enabled
        self.current_env = None
        self.proofs_in_progress: Dict[str, ProofInProgress] = {}

        # Configure the server with mathlib and aesop
        if mathlib_enabled:
            config = LeanREPLConfig(
                project=TempRequireProject(
                    require="mathlib",
                )
            )
        else:
            config = LeanREPLConfig()

        self.server = AutoLeanServer(config)

    def run(self, lean_code: str) -> ProofResult:
        """Execute Lean code and return the result."""
        try:
            response = self.server.run(Command(cmd=lean_code, env=self.current_env))

            if isinstance(response, LeanError):
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=str(response.message),
                    error=str(response.message)
                )

            # Update environment
            self.current_env = response.env

            # Check for sorries
            has_sorry = len(response.sorries) > 0

            # Format response and determine if there are actual errors (not just warnings)
            has_actual_errors = False
            if response.messages:
                formatted_messages = []
                for msg in response.messages:
                    formatted_messages.append(f"[{msg.severity}] {msg.data}")
                    if msg.severity == 'error':
                        has_actual_errors = True
                response_text = "\n".join(formatted_messages)
            else:
                response_text = "Success"

            # Success means no actual errors (warnings are OK)
            success = not has_actual_errors

            # Check if proof is complete (no errors and no sorries)
            proof_complete = success and not has_sorry

            # Get proof state from sorry if available
            proof_state = None
            if response.sorries:
                proof_state = response.sorries[0].proof_state

            return ProofResult(
                success=success,
                proof_complete=proof_complete,
                has_sorry=has_sorry,
                response=response_text,
                proof_state=proof_state,
                error=response_text if has_actual_errors else None
            )

        except Exception as e:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=str(e),
                error=str(e)
            )

    def run_command(self, command: str, env: Optional[int] = None) -> ProofResult:
        """Execute a standalone Lean command."""
        try:
            response = self.server.run(Command(cmd=command, env=env or self.current_env))

            if isinstance(response, LeanError):
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=str(response.message),
                    error=str(response.message)
                )

            # Update environment
            self.current_env = response.env

            # Check for sorries
            has_sorry = len(response.sorries) > 0

            # Format response and determine if there are actual errors (not just warnings)
            has_actual_errors = False
            if response.messages:
                formatted_messages = []
                for msg in response.messages:
                    formatted_messages.append(f"[{msg.severity}] {msg.data}")
                    if msg.severity == 'error':
                        has_actual_errors = True
                response_text = "\n".join(formatted_messages)
            else:
                response_text = "Success"

            # Success means no actual errors (warnings are OK)
            success = not has_actual_errors

            # Check if proof is complete (no errors and no sorries)
            proof_complete = success and not has_sorry

            # Get proof state from sorry if available
            proof_state = None
            if response.sorries:
                proof_state = response.sorries[0].proof_state

            return ProofResult(
                success=success,
                proof_complete=proof_complete,
                has_sorry=has_sorry,
                response=response_text,
                proof_state=proof_state,
                error=response_text if has_actual_errors else None
            )

        except Exception as e:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=str(e),
                error=str(e)
            )

    def run_proof_step(self, proof_state: int, tactic: str) -> ProofResult:
        """Execute a proof step on a given proof state."""
        try:
            response = self.server.run(ProofStep(proof_state=proof_state, tactic=tactic))

            if isinstance(response, LeanError):
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=str(response.message),
                    error=str(response.message)
                )

            # Check if proof is complete
            proof_complete = response.proof_status == "Complete"

            # Get new proof state and goals
            new_proof_state = response.proof_state
            goals = response.goals if hasattr(response, 'goals') else []

            return ProofResult(
                success=True,
                proof_complete=proof_complete,
                has_sorry=False,
                response=response.proof_status,
                proof_state=new_proof_state,
                goals=goals
            )

        except Exception as e:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=str(e),
                error=str(e)
            )

    def start_proof(self, name: str, statement: str) -> ProofResult:
        """Start a new proof with 'by sorry' to get the initial proof state."""
        full_theorem = f"theorem {name} {statement} := by sorry"

        result = self.run_command(full_theorem)

        # Check if we have a proof state, even if there are warnings
        if result.proof_state is not None:
            # Get the initial goals by running a skip tactic
            initial_step = self.run_proof_step(result.proof_state, "skip")

            if initial_step.success:
                self.proofs_in_progress[name] = ProofInProgress(
                    name=name,
                    statement=statement,
                    initial_proof_state=result.proof_state,
                    current_proof_state=result.proof_state,
                    tactic_history=[],
                    goals=initial_step.goals or []
                )

                return ProofResult(
                    success=True,
                    proof_complete=False,
                    has_sorry=True,
                    response=f"Started proof '{name}' with initial state {result.proof_state}",
                    proof_state=result.proof_state,
                    goals=initial_step.goals
                )

        return result

    def apply_tactic_to_proof(self, proof_name: str, tactic: str) -> ProofResult:
        """Apply a tactic to a proof in progress."""
        if proof_name not in self.proofs_in_progress:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"No proof named '{proof_name}' in progress",
                error=f"No proof named '{proof_name}' in progress"
            )

        proof = self.proofs_in_progress[proof_name]

        if proof.completed:
            return ProofResult(
                success=False,
                proof_complete=True,
                has_sorry=False,
                response=f"Proof '{proof_name}' is already completed",
                error=f"Proof '{proof_name}' is already completed"
            )

        # Apply the tactic
        result = self.run_proof_step(proof.current_proof_state, tactic)

        if result.success:
            # Update the proof state
            proof.tactic_history.append((proof.current_proof_state, tactic))
            proof.current_proof_state = result.proof_state
            proof.goals = result.goals or []

            if result.proof_complete:
                proof.completed = True

        return result

    def get_proof_state(self, proof_name: str) -> Optional[ProofInProgress]:
        """Get the current state of a proof in progress."""
        return self.proofs_in_progress.get(proof_name)

    def backtrack_proof(self, proof_name: str, steps: int = 1) -> ProofResult:
        """Backtrack a proof by removing the last n steps."""
        if proof_name not in self.proofs_in_progress:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"No proof named '{proof_name}' in progress",
                error=f"No proof named '{proof_name}' in progress"
            )

        proof = self.proofs_in_progress[proof_name]

        if len(proof.tactic_history) < steps:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"Cannot backtrack {steps} steps, only {len(proof.tactic_history)} steps in history",
                error=f"Cannot backtrack {steps} steps, only {len(proof.tactic_history)} steps in history"
            )

        # Remove the last steps
        for _ in range(steps):
            proof.tactic_history.pop()

        # Rebuild the proof state by replaying tactics
        if proof.tactic_history:
            current_state = proof.initial_proof_state
            for from_state, tactic in proof.tactic_history:
                step_result = self.run_proof_step(current_state, tactic)
                if step_result.success:
                    current_state = step_result.proof_state
                else:
                    return ProofResult(
                        success=False,
                        proof_complete=False,
                        has_sorry=False,
                        response=f"Failed to replay tactic: {step_result.error}",
                        error=f"Failed to replay tactic: {step_result.error}"
                    )

            proof.current_proof_state = current_state
            # Get current goals
            current_step = self.run_proof_step(current_state, "skip")
            if current_step.success:
                proof.goals = current_step.goals or []
        else:
            # Back to initial state
            proof.current_proof_state = proof.initial_proof_state
            initial_step = self.run_proof_step(proof.initial_proof_state, "skip")
            if initial_step.success:
                proof.goals = initial_step.goals or []

        proof.completed = False

        return ProofResult(
            success=True,
            proof_complete=False,
            has_sorry=True,
            response=f"Backtracked {steps} steps",
            proof_state=proof.current_proof_state,
            goals=proof.goals
        )

    def get_current_env(self) -> Optional[int]:
        """Get the current environment ID."""
        return self.current_env

    def set_current_env(self, env: int):
        """Set the current environment ID."""
        self.current_env = env

    def run_multi_step(self, proof_state: int, tactics: List[str]) -> List[ProofResult]:
        """Run multiple proof steps in sequence."""
        results = []
        current_state = proof_state

        for tactic in tactics:
            result = self.run_proof_step(current_state, tactic)
            results.append(result)

            if result.success and result.proof_state is not None:
                current_state = result.proof_state
            else:
                # Stop on first failure
                break

        return results
