# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Lean Interact Session Manager

This module provides a stateful interface for interactive Lean proof development
using the official LeanInteract library. It maintains proof states, supports branching,
and provides comprehensive proof session management.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# LeanInteract imports
from lean_interact import LeanServer, AutoLeanServer, LeanREPLConfig, TemporaryProject, TempRequireProject, LeanRequire
from lean_interact import Command, ProofStep, FileCommand
from lean_interact.interface import InfoTree, Message

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

class ProofStatus(Enum):
    """Status of a proof branch"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ProofStepInfo:
    """A single step in a proof"""
    step_id: str
    tactic: str
    result: Dict[str, Any]
    timestamp: str
    goals_before: List[str]
    goals_after: List[str]
    success: bool
    error: Optional[str] = None
    proof_state_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_id": self.step_id,
            "tactic": self.tactic,
            "result": self.result,
            "timestamp": self.timestamp,
            "goals_before": self.goals_before,
            "goals_after": self.goals_after,
            "success": self.success,
            "error": self.error,
            "proof_state_id": self.proof_state_id
        }

@dataclass
class ProofBranch:
    """A branch in the proof tree"""
    branch_id: str
    name: str
    parent_branch: Optional[str]
    steps: List[ProofStepInfo] = field(default_factory=list)
    current_goals: List[str] = field(default_factory=list)
    status: ProofStatus = ProofStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_proof_state: Optional[int] = None
    current_env: Optional[int] = None

    def add_step(self, step: ProofStepInfo):
        """Add a step to this branch"""
        self.steps.append(step)
        if step.success:
            self.current_goals = step.goals_after
            self.current_proof_state = step.proof_state_id
            if not step.goals_after:  # No remaining goals
                self.status = ProofStatus.COMPLETED
        else:
            self.status = ProofStatus.FAILED

class LeanInteractSession:
    """
    A stateful Lean proof session using the official LeanInteract library.
    Manages proof states, branching, and interactive proof development.
    """

    def __init__(self, theorem_statement: str, repo_path: Optional[str] = None,
                 timeout: float = 30.0, lean_path: Optional[str] = None,
                 use_mathlib: bool = True, use_auto_server: bool = True):
        """
        Initialize a Lean proof session.

        Args:
            theorem_statement: The theorem to prove
            repo_path: Optional path to Lean repository
            timeout: Default timeout for Lean execution
            lean_path: Path to Lean executable (ignored, LeanInteract manages this)
            use_mathlib: Whether to include Mathlib imports (default: True, uses TempRequireProject for reliability)
            use_auto_server: Whether to use AutoLeanServer for better recovery
        """
        self.session_id = str(uuid.uuid4())
        self.theorem_statement = theorem_statement
        self.repo_path = repo_path
        self.timeout = timeout
        self.use_mathlib = use_mathlib
        self.use_auto_server = use_auto_server
        self.created_at = datetime.now().isoformat()

        # Initialize branches
        self.branches: Dict[str, ProofBranch] = {}
        self.current_branch_id = self._create_main_branch()

        # Initialize LeanInteract components
        self._setup_lean_server()
        self._initialize_proof_state()

    def _setup_lean_server(self):
        """Setup the LeanInteract server and configuration"""
        try:
            if self.use_mathlib:
                # Use TempRequireProject with mathlib and aesop
                # Mathlib typically includes aesop as a dependency
                project = TempRequireProject(require="mathlib")
                self.config = LeanREPLConfig(project=project)
            else:
                # Use default configuration without mathlib
                self.config = LeanREPLConfig()

            # Create the server
            if self.use_auto_server:
                self.server = AutoLeanServer(self.config)
            else:
                self.server = LeanServer(self.config)

            LOG.info(f"Created Lean server with mathlib={self.use_mathlib}")

        except Exception as e:
            LOG.error(f"Failed to setup Lean server: {e}")
            raise

    def _create_main_branch(self) -> str:
        """Create the main proof branch"""
        branch_id = str(uuid.uuid4())

        main_branch = ProofBranch(
            branch_id=branch_id,
            name="main",
            parent_branch=None,
            current_goals=["Initial goal"] # Will be updated when we initialize
        )

        self.branches[branch_id] = main_branch
        return branch_id

    def _initialize_proof_state(self):
        """Initialize the proof state with the theorem"""
        try:
            # Extract the theorem signature (everything before := by)
            if self.use_mathlib:
                # Add mathlib imports - TempRequireProject makes these available
                imports = """
import Mathlib
import Aesop

open BigOperators Real Nat

"""
                # Extract theorem signature without the proof
                import re
                # Match theorem signature up to ":=" but exclude the proof part
                match = re.match(r'(theorem\s+[^:]+:\s*[^:]+)\s*:=\s*by.*', self.theorem_statement, re.DOTALL)
                if match:
                    theorem_signature = match.group(1).strip()
                    # Create incomplete theorem declaration
                    theorem_decl = f"{theorem_signature} := by sorry"
                else:
                    # Fallback: just replace the proof with sorry
                    theorem_decl = re.sub(r':=\s*by.*', ':= by sorry', self.theorem_statement, flags=re.DOTALL)

                full_command = imports + theorem_decl
            else:
                # Same logic without mathlib
                import re
                match = re.match(r'(theorem\s+[^:]+:\s*[^:]+)\s*:=\s*by.*', self.theorem_statement, re.DOTALL)
                if match:
                    theorem_signature = match.group(1).strip()
                    theorem_decl = f"{theorem_signature} := by sorry"
                else:
                    theorem_decl = re.sub(r':=\s*by.*', ':= by sorry', self.theorem_statement, flags=re.DOTALL)
                full_command = theorem_decl

            # Execute the command to set up the theorem
            response = self.server.run(Command(cmd=full_command))

            # Update current branch with initial state
            current_branch = self.branches[self.current_branch_id]
            current_branch.current_env = response.env

            # Extract goals from response
            if hasattr(response, 'goals') and response.goals:
                goals = [str(goal.target) for goal in response.goals]
                current_branch.current_goals = goals
                # Get the proof state ID
                current_branch.current_proof_state = response.goals[0].proof_state if response.goals else None
                LOG.info(f"Initialized proof state with {len(current_branch.current_goals)} goals")
            else:
                # If no goals from Command, extract from theorem signature
                import re
                # Extract just the goal part (after the last colon before :=)
                match = re.search(r':\s*([^:]+)\s*:=', self.theorem_statement)
                if match:
                    goal_text = match.group(1).strip()
                    current_branch.current_goals = [goal_text]
                    current_branch.current_proof_state = None  # Will be set when first tactic is applied
                    LOG.info(f"Extracted goal from theorem statement: {goal_text}")
                else:
                    current_branch.current_goals = ["Unknown goal"]
                    current_branch.current_proof_state = None
                    LOG.warning("Could not extract goal from theorem statement")

        except Exception as e:
            LOG.error(f"Failed to initialize proof state: {e}")
            current_branch = self.branches[self.current_branch_id]
            current_branch.status = ProofStatus.FAILED
            current_branch.current_goals = [f"Error: {str(e)}"]

    def apply_tactic(self, tactic: str) -> Dict[str, Any]:
        """
        Apply a tactic to the current proof state.

        Args:
            tactic: The tactic to apply

        Returns:
            Dictionary with success status, goals, and other information
        """
        try:
            current_branch = self.branches[self.current_branch_id]

            if current_branch.status != ProofStatus.ACTIVE:
                return {
                    "success": False,
                    "error": f"Branch is not active (status: {current_branch.status.value})",
                    "goals": current_branch.current_goals,
                    "proof_complete": False
                }

            if not current_branch.current_proof_state:
                # First tactic - need to initialize proof state
                import re

                # Extract theorem signature and replace sorry with the new tactic
                if self.use_mathlib:
                    imports = """
import Mathlib
import Aesop

open BigOperators Real Nat

"""
                    # Extract theorem signature and replace with new tactic
                    match = re.match(r'(theorem\s+[^:]+:\s*[^:]+)\s*:=\s*by.*', self.theorem_statement, re.DOTALL)
                    if match:
                        theorem_signature = match.group(1).strip()
                        theorem_with_tactic = f"{theorem_signature} := by {tactic}"
                    else:
                        theorem_with_tactic = re.sub(r':=\s*by.*', f':= by {tactic}', self.theorem_statement, flags=re.DOTALL)

                    full_command = imports + theorem_with_tactic
                else:
                    # Same logic without mathlib
                    match = re.match(r'(theorem\s+[^:]+:\s*[^:]+)\s*:=\s*by.*', self.theorem_statement, re.DOTALL)
                    if match:
                        theorem_signature = match.group(1).strip()
                        theorem_with_tactic = f"{theorem_signature} := by {tactic}"
                    else:
                        theorem_with_tactic = re.sub(r':=\s*by.*', f':= by {tactic}', self.theorem_statement, flags=re.DOTALL)

                    full_command = theorem_with_tactic

                # Execute the full theorem with the tactic
                response = self.server.run(Command(cmd=full_command))

                # Check if the tactic succeeded
                success = self._is_response_successful(response)

                if success:
                    # Extract goals - could be from response.goals or from "unsolved goals" message
                    goals_after, new_proof_state = self._extract_goals_from_response(response)
                    proof_complete = len(goals_after) == 0
                else:
                    goals_after = current_branch.current_goals.copy()
                    new_proof_state = None
                    proof_complete = False
            else:
                # Apply the tactic to existing proof state
                response = self.server.run(ProofStep(
                    proof_state=current_branch.current_proof_state,
                    tactic=tactic
                ))

                # Process the response
                success = self._is_response_successful(response)

                if success:
                    # Extract goals - could be from response.goals or from "unsolved goals" message
                    goals_after, new_proof_state = self._extract_goals_from_response(response)
                    proof_complete = len(goals_after) == 0
                else:
                    goals_after = []
                    new_proof_state = None
                    proof_complete = False

            # Common processing for both paths
            goals_before = current_branch.current_goals.copy()

            # Create step record
            step = ProofStepInfo(
                step_id=str(uuid.uuid4()),
                tactic=tactic,
                result={
                    "messages": [{"severity": msg.severity, "data": msg.data} for msg in response.messages] if response.messages else [],
                    "goals": goals_after,
                    "proof_state": new_proof_state
                },
                timestamp=datetime.now().isoformat(),
                goals_before=goals_before,
                goals_after=goals_after,
                success=success,
                error=None if success else "Tactic failed",
                proof_state_id=new_proof_state
            )

            # Update branch
            current_branch.add_step(step)

            # Check if proof is complete
            proof_complete = success and len(goals_after) == 0

            # Additional validation: if we claim the proof is complete,
            # ensure the proof script doesn't contain sorry
            if proof_complete:
                temp_script = self._generate_proof_script_unsafe()
                if 'sorry' in temp_script:
                    LOG.warning(f"Proof reported as complete but script contains sorry: {temp_script}")
                    proof_complete = False
                    success = False

            return {
                "success": success,
                "goals": goals_after,
                "proof_complete": proof_complete,
                "step_id": step.step_id,
                "messages": step.result.get("messages", []),
                "error": step.error
            }

        except Exception as e:
            LOG.error(f"Error applying tactic '{tactic}': {e}")

            # Create failed step record
            step = ProofStepInfo(
                step_id=str(uuid.uuid4()),
                tactic=tactic,
                result={"error": str(e)},
                timestamp=datetime.now().isoformat(),
                goals_before=current_branch.current_goals.copy(),
                goals_after=current_branch.current_goals.copy(),
                success=False,
                error=str(e),
                proof_state_id=current_branch.current_proof_state
            )

            current_branch.add_step(step)

            return {
                "success": False,
                "goals": current_branch.current_goals,
                "proof_complete": False,
                "step_id": step.step_id,
                "messages": [],
                "error": str(e)
            }

    def get_proof_state(self) -> Dict[str, Any]:
        """Get the current proof state"""
        current_branch = self.branches[self.current_branch_id]

        return {
            "session_id": self.session_id,
            "branch_id": self.current_branch_id,
            "branch_name": current_branch.name,
            "current_goals": current_branch.current_goals,
            "status": current_branch.status.value,
            "step_count": len(current_branch.steps),
            "proof_complete": current_branch.status == ProofStatus.COMPLETED,
            "last_step": current_branch.steps[-1].to_dict() if current_branch.steps else None
        }

    def create_branch(self, name: str, from_step: Optional[str] = None) -> str:
        """
        Create a new branch for alternative proof exploration.

        Args:
            name: Name for the new branch
            from_step: Optional step ID to branch from (defaults to current state)

        Returns:
            New branch ID
        """
        current_branch = self.branches[self.current_branch_id]
        branch_id = str(uuid.uuid4())

        # Find the step to branch from
        if from_step:
            step_index = None
            for i, step in enumerate(current_branch.steps):
                if step.step_id == from_step:
                    step_index = i
                    break

            if step_index is None:
                raise ValueError(f"Step {from_step} not found in current branch")
        else:
            step_index = len(current_branch.steps) - 1

        # Create new branch
        new_branch = ProofBranch(
            branch_id=branch_id,
            name=name,
            parent_branch=self.current_branch_id
        )

        # Copy state up to the branch point
        if step_index >= 0:
            new_branch.steps = current_branch.steps[:step_index + 1].copy()
            last_step = current_branch.steps[step_index]
            new_branch.current_goals = last_step.goals_after.copy()
            new_branch.current_proof_state = last_step.proof_state_id
        else:
            new_branch.current_goals = current_branch.current_goals.copy()
            new_branch.current_proof_state = current_branch.current_proof_state

        new_branch.current_env = current_branch.current_env

        self.branches[branch_id] = new_branch

        LOG.info(f"Created branch '{name}' with ID {branch_id}")
        return branch_id

    def switch_branch(self, branch_id: str):
        """Switch to a different branch"""
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} does not exist")

        self.current_branch_id = branch_id
        LOG.info(f"Switched to branch {self.branches[branch_id].name}")

    def list_branches(self) -> List[Dict[str, Any]]:
        """List all branches"""
        return [
            {
                "branch_id": branch_id,
                "name": branch.name,
                "status": branch.status.value,
                "step_count": len(branch.steps),
                "current_goals": len(branch.current_goals),
                "is_current": branch_id == self.current_branch_id
            }
            for branch_id, branch in self.branches.items()
        ]

    def undo_step(self, steps: int = 1) -> Dict[str, Any]:
        """
        Undo the last N steps in the current branch.

        Args:
            steps: Number of steps to undo (default: 1)

        Returns:
            Dictionary with success status and new state
        """
        current_branch = self.branches[self.current_branch_id]

        if len(current_branch.steps) < steps:
            return {
                "success": False,
                "error": f"Cannot undo {steps} steps, only {len(current_branch.steps)} steps available"
            }

        # Remove the last N steps
        undone_steps = current_branch.steps[-steps:]
        current_branch.steps = current_branch.steps[:-steps]

        # Restore state
        if current_branch.steps:
            last_step = current_branch.steps[-1]
            current_branch.current_goals = last_step.goals_after.copy()
            current_branch.current_proof_state = last_step.proof_state_id
        else:
            # Back to initial state
            current_branch.current_goals = ["Initial goal"]  # Would need to re-initialize properly
            current_branch.current_proof_state = None

        # Update status
        current_branch.status = ProofStatus.ACTIVE

        return {
            "success": True,
            "undone_steps": [step.to_dict() for step in undone_steps],
            "current_goals": current_branch.current_goals,
            "step_count": len(current_branch.steps)
        }

    def _generate_proof_script_unsafe(self) -> str:
        """Generate the proof script without validation (internal use)"""
        current_branch = self.branches[self.current_branch_id]

        if not current_branch.steps:
            return self.theorem_statement

        # Extract successful tactics
        tactics = []
        for step in current_branch.steps:
            if step.success:
                tactics.append(step.tactic)

        if not tactics:
            return self.theorem_statement

        # Build proof script - handle both single line and multi-line sorry patterns
        base_theorem = self.theorem_statement

        # Use regex to find and replace sorry patterns more robustly
        import re

        # Pattern to match "by sorry" or "by\n  sorry" or "by\n    sorry" etc.
        by_sorry_pattern = r'by\s+sorry'

        # Pattern to match " := by sorry" or " := by\n  sorry" etc.
        assign_by_sorry_pattern = r':=\s+by\s+sorry'

        if len(tactics) == 1:
            # Single tactic - replace inline
            if re.search(by_sorry_pattern, base_theorem):
                proof_script = re.sub(by_sorry_pattern, f'by {tactics[0]}', base_theorem)
            elif re.search(assign_by_sorry_pattern, base_theorem):
                proof_script = re.sub(assign_by_sorry_pattern, f':= by {tactics[0]}', base_theorem)
            else:
                proof_script = base_theorem + f"\n-- Applied tactics: {'; '.join(tactics)}"
        else:
            # Multiple tactics - use block format
            tactic_block = "\n  ".join(tactics)
            if re.search(by_sorry_pattern, base_theorem):
                proof_script = re.sub(by_sorry_pattern, f'by\n  {tactic_block}', base_theorem)
            elif re.search(assign_by_sorry_pattern, base_theorem):
                proof_script = re.sub(assign_by_sorry_pattern, f':= by\n  {tactic_block}', base_theorem)
            else:
                proof_script = base_theorem + f"\n-- Applied tactics: {'; '.join(tactics)}"

        return proof_script

    def get_proof_script(self) -> str:
        """Generate the proof script for the current branch"""
        current_branch = self.branches[self.current_branch_id]

        # Generate the proof script
        proof_script = self._generate_proof_script_unsafe()

        # Validate that sorry has been properly removed
        if 'sorry' in proof_script:
            LOG.warning(f"Generated proof script still contains 'sorry': {proof_script}")
            # This indicates the proof is not actually valid
            current_branch.status = ProofStatus.FAILED
            return f"-- ERROR: Proof script still contains 'sorry'\n{proof_script}"

        return proof_script

    def analyze_proof_attempts(self) -> Dict[str, Any]:
        """Analyze proof attempts across all branches"""
        total_steps = sum(len(branch.steps) for branch in self.branches.values())
        successful_steps = sum(
            len([step for step in branch.steps if step.success])
            for branch in self.branches.values()
        )

        completed_branches = len([
            branch for branch in self.branches.values()
            if branch.status == ProofStatus.COMPLETED
        ])

        return {
            "total_branches": len(self.branches),
            "completed_branches": completed_branches,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0.0,
            "branches": {
                branch_id: {
                    "name": branch.name,
                    "status": branch.status.value,
                    "steps": len(branch.steps),
                    "goals": len(branch.current_goals)
                }
                for branch_id, branch in self.branches.items()
            }
        }

    def save_session(self, filepath: str) -> bool:
        """Save the session to a file"""
        try:
            session_data = {
                "session_id": self.session_id,
                "theorem_statement": self.theorem_statement,
                "created_at": self.created_at,
                "use_mathlib": self.use_mathlib,
                "current_branch_id": self.current_branch_id,
                "branches": {
                    branch_id: {
                        "branch_id": branch.branch_id,
                        "name": branch.name,
                        "parent_branch": branch.parent_branch,
                        "steps": [step.to_dict() for step in branch.steps],
                        "current_goals": branch.current_goals,
                        "status": branch.status.value,
                        "created_at": branch.created_at
                    }
                    for branch_id, branch in self.branches.items()
                }
            }

            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)

            LOG.info(f"Session saved to {filepath}")
            return True

        except Exception as e:
            LOG.error(f"Failed to save session: {e}")
            return False

    @classmethod
    def load_session(cls, filepath: str) -> 'LeanInteractSession':
        """Load a session from a file"""
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            # Create new session
            session = cls(
                theorem_statement=session_data["theorem_statement"],
                use_mathlib=session_data.get("use_mathlib", True)
            )

            # Restore session data
            session.session_id = session_data["session_id"]
            session.created_at = session_data["created_at"]
            session.current_branch_id = session_data["current_branch_id"]

            # Restore branches
            session.branches = {}
            for branch_id, branch_data in session_data["branches"].items():
                branch = ProofBranch(
                    branch_id=branch_data["branch_id"],
                    name=branch_data["name"],
                    parent_branch=branch_data["parent_branch"],
                    current_goals=branch_data["current_goals"],
                    status=ProofStatus(branch_data["status"]),
                    created_at=branch_data["created_at"]
                )

                # Restore steps
                for step_data in branch_data["steps"]:
                    step = ProofStepInfo(
                        step_id=step_data["step_id"],
                        tactic=step_data["tactic"],
                        result=step_data["result"],
                        timestamp=step_data["timestamp"],
                        goals_before=step_data["goals_before"],
                        goals_after=step_data["goals_after"],
                        success=step_data["success"],
                        error=step_data.get("error"),
                        proof_state_id=step_data.get("proof_state_id")
                    )
                    branch.steps.append(step)

                session.branches[branch_id] = branch

            LOG.info(f"Session loaded from {filepath}")
            return session

        except Exception as e:
            LOG.error(f"Failed to load session: {e}")
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'server'):
                # The LeanInteract server should handle cleanup automatically
                pass
            LOG.info(f"Session {self.session_id} cleaned up")
        except Exception as e:
            LOG.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

    def _is_response_successful(self, response: Any) -> bool:
        """
        Helper to check if a LeanInteract response indicates success.
        Treats 'declaration uses sorry' warnings as failures.
        Treats 'unsolved goals' as success (partial progress).
        """
        if not response.messages:
            return True

        # Check for 'error' severity messages
        for msg in response.messages:
            if msg.severity == 'error':
                # "unsolved goals" is actually partial progress, not failure
                if "unsolved goals" in msg.data:
                    return True
                # Real errors that indicate failure
                return False

        # Check for 'warning' messages that indicate proof invalidity
        for msg in response.messages:
            if msg.severity == 'warning':
                # Check if the warning is about using 'sorry'
                if "declaration uses 'sorry'" in msg.data:
                    return False
                # Add other warning patterns that should be treated as failures
                if "sorry" in msg.data.lower():
                    return False

        return True

    def _extract_goals_from_response(self, response: Any) -> Tuple[List[str], Optional[int]]:
        """
        Extract goals from a LeanInteract response.
        Handles both response.goals and "unsolved goals" messages.
        """
        goals_after = []
        new_proof_state = None

        if hasattr(response, 'goals') and response.goals:
            goals_after = [str(goal.target) for goal in response.goals]
            new_proof_state = response.goals[0].proof_state if response.goals else None
        elif response.messages:
            for msg in response.messages:
                if msg.severity == 'error' and 'unsolved goals' in msg.data:
                    # Extract goals from the "unsolved goals" message
                    import re
                    # Find all lines that start with ⊢ (the goal symbol)
                    goal_lines = re.findall(r'⊢\s*(.+)', msg.data)
                    goals_after = [goal.strip() for goal in goal_lines]
                    # No proof_state available from error messages
                    new_proof_state = None
                    break

        return goals_after, new_proof_state


# Factory function to create LeanInteractSession instances
def create_lean_interact_session(theorem_statement: str, repo_path: Optional[str] = None,
                                timeout: float = 30.0, lean_path: Optional[str] = None,
                                use_mathlib: bool = True, use_auto_server: bool = True) -> LeanInteractSession:
    """
    Create a new Lean interact session.

    Args:
        theorem_statement: The theorem to prove
        repo_path: Optional path to Lean repository
        timeout: Default timeout for Lean execution
        lean_path: Path to Lean executable (ignored, LeanInteract manages this)
        use_mathlib: Whether to include Mathlib imports (default: True, uses TempRequireProject for reliability)
        use_auto_server: Whether to use AutoLeanServer for better recovery

    Returns:
        LeanInteractSession instance
    """
    return LeanInteractSession(theorem_statement, repo_path, timeout, lean_path, use_mathlib, use_auto_server)
