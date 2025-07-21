#!/usr/bin/env python3
"""
Thread-Safe Lean 4 Prover

This module provides a thread-safe Lean 4 prover that allows multiple agents to solve
different proofs concurrently without interfering with each other's solving process.

Key Features:
- Per-thread isolation of proof state
- Thread-safe proof management
- Concurrent agent instances that don't interfere
- Safe for thousands of concurrent agents
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
import copy

from lean_interact import AutoLeanServer, LeanREPLConfig, Command, ProofStep, TempRequireProject
from lean_interact.interface import CommandResponse, ProofStepResponse, LeanError


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
    """
    Thread-safe Lean 4 prover that ensures concurrent agents don't interfere.

    Key Design:
    - Each method call is atomic
    - Environment state is properly isolated
    - Proof state management is thread-safe
    - Multiple instances can run concurrently
    """

    def __init__(self, mathlib_enabled: bool = True):
        """Initialize thread-safe prover."""
        self._mathlib_enabled = mathlib_enabled
        self._lock = threading.RLock()  # Reentrant lock
        self._thread_local = threading.local()

        # Thread-safe proof registry
        self._proofs_registry: Dict[str, ProofInProgress] = {}
        self._proofs_lock = threading.RLock()

        # Instance ID for debugging
        self._instance_id = str(uuid.uuid4())[:8]

    def _get_server(self) -> AutoLeanServer:
        """Get or create thread-local server instance."""
        if not hasattr(self._thread_local, 'server'):
            # Each thread gets its own AutoLeanServer instance
            if self._mathlib_enabled:
                config = LeanREPLConfig(
                    project=TempRequireProject(
                        require="mathlib",
                    )
                )
            else:
                config = LeanREPLConfig()

            self._thread_local.server = AutoLeanServer(config)
            self._thread_local.current_env = None
            self._thread_local.proofs_in_progress = {}
            self._thread_local.thread_id = threading.get_ident()

        return self._thread_local.server

    @property
    def mathlib_enabled(self) -> bool:
        """Check if mathlib is enabled."""
        return self._mathlib_enabled

    @property
    def current_env(self) -> Optional[int]:
        """Get current environment for this thread."""
        server = self._get_server()
        return getattr(self._thread_local, 'current_env', None)

    @property
    def proofs_in_progress(self) -> Dict[str, ProofInProgress]:
        """Get proofs in progress for this thread."""
        server = self._get_server()  # Ensure thread local is initialized
        return getattr(self._thread_local, 'proofs_in_progress', {})

    def run(self, lean_code: str) -> ProofResult:
        """Thread-safe execution of Lean code."""
        server = self._get_server()

        try:
            response = server.run(Command(cmd=lean_code, env=self._thread_local.current_env))

            if isinstance(response, LeanError):
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=str(response.message),
                    error=str(response.message)
                )

            # Update environment
            self._thread_local.current_env = response.env

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
        """Thread-safe command execution."""
        server = self._get_server()

        try:
            response = server.run(Command(cmd=command, env=env or self._thread_local.current_env))

            if isinstance(response, LeanError):
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=str(response.message),
                    error=str(response.message)
                )

            # Update environment
            self._thread_local.current_env = response.env

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
        """Thread-safe proof step execution."""
        server = self._get_server()

        try:
            response = server.run(ProofStep(proof_state=proof_state, tactic=tactic))

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
        """Thread-safe proof initialization."""
        # Generate unique proof name per thread to avoid conflicts
        thread_id = threading.get_ident()
        unique_name = f"{name}_thread_{thread_id}_{int(time.time() * 1000) % 10000}"

        full_theorem = f"theorem {unique_name} {statement} := by sorry"
        result = self.run_command(full_theorem)

        # Check if we have a proof state, even if there are warnings
        if result.proof_state is not None:
            # Get the initial goals by running a skip tactic
            initial_step = self.run_proof_step(result.proof_state, "skip")

            if initial_step.success:
                proof_obj = ProofInProgress(
                    name=name,
                    statement=statement,
                    initial_proof_state=result.proof_state,
                    current_proof_state=result.proof_state,
                    tactic_history=[],
                    goals=initial_step.goals or []
                )

                # Store in thread-local storage
                self._thread_local.proofs_in_progress[name] = proof_obj

                # Also store in global registry for cross-thread access
                with self._proofs_lock:
                    self._proofs_registry[name] = copy.deepcopy(proof_obj)

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
        """Thread-safe tactic application."""
        proofs = self.proofs_in_progress

        if proof_name not in proofs:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"No proof named '{proof_name}' in progress",
                error=f"No proof named '{proof_name}' in progress"
            )

        proof = proofs[proof_name]

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

            # Update global registry
            with self._proofs_lock:
                self._proofs_registry[proof_name] = copy.deepcopy(proof)

        return result

    def get_proof_state(self, proof_name: str) -> Optional[ProofInProgress]:
        """Thread-safe proof state retrieval."""
        proofs = self.proofs_in_progress
        if proof_name in proofs:
            return copy.deepcopy(proofs[proof_name])
        return None

    def backtrack_proof(self, proof_name: str, steps: int = 1) -> ProofResult:
        """Thread-safe proof backtracking."""
        proofs = self.proofs_in_progress

        if proof_name not in proofs:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"No proof named '{proof_name}' in progress",
                error=f"No proof named '{proof_name}' in progress"
            )

        proof = proofs[proof_name]

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

        # Update global registry
        with self._proofs_lock:
            self._proofs_registry[proof_name] = copy.deepcopy(proof)

        return ProofResult(
            success=True,
            proof_complete=False,
            has_sorry=True,
            response=f"Backtracked {steps} steps",
            proof_state=proof.current_proof_state,
            goals=proof.goals
        )

    def get_current_env(self) -> Optional[int]:
        """Get the current environment ID for this thread."""
        return getattr(self._thread_local, 'current_env', None)

    def set_current_env(self, env: int):
        """Set the current environment ID for this thread."""
        self._get_server()  # Ensure thread local is initialized
        self._thread_local.current_env = env

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

    def get_thread_info(self) -> Dict[str, Any]:
        """Get debugging info about current thread's prover state."""
        thread_id = threading.get_ident()
        proofs = self.proofs_in_progress

        return {
            'instance_id': self._instance_id,
            'thread_id': thread_id,
            'current_env': self.get_current_env(),
            'active_proofs': list(proofs.keys()),
            'thread_local_proofs': len(proofs),
            'global_registry_proofs': len(self._proofs_registry)
        }
