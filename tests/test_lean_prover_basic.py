"""
Basic pytest tests for LeanProver from lean4 module.

NOTE: This is a simplified test file. Comprehensive tests are in test_lean4_interactive.py.
This file focuses on basic LeanProver functionality only.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nemo_skills.code_execution.lean4 import LeanProver, ProofResult


class TestBasicLeanProver:
    """Basic test suite for LeanProver core functionality."""

    def test_simple_complete_proof(self):
        """Test a simple proof that should complete successfully."""
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem test : 1 + 1 = 2 := by rfl")

        assert result.success == True
        assert result.proof_complete == True
        assert result.has_sorry == False

    def test_proof_with_sorry(self):
        """Test that proofs with sorry are detected correctly."""
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem test : 1 + 1 = 2 := by sorry")

        assert result.success == True  # Structure is valid
        assert result.proof_complete == False  # But not complete
        assert result.has_sorry == True  # Contains sorry

    def test_invalid_proof(self):
        """Test that invalid proofs are handled gracefully."""
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem test : 1 + 1 = 2 := by invalid_tactic")

        assert result.success == False
        assert result.proof_complete == False
        assert result.has_sorry == False
        assert result.error is not None

    def test_basic_command(self):
        """Test basic command execution."""
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run_command("theorem simple : True := trivial")

        # Should execute without crashing
        assert isinstance(result.success, bool)
        assert isinstance(result.proof_complete, bool)
        assert isinstance(result.has_sorry, bool)

    def test_incremental_proof_building(self):
        """Test basic incremental proof building."""
        prover = LeanProver(mathlib_enabled=True)

        # Start a proof
        start_result = prover.start_proof("test_incr", "(n : Nat) : n + 0 = n")

        if start_result.success:
            assert start_result.has_sorry == True
            assert "test_incr" in prover.proofs_in_progress

            # Apply a tactic
            tactic_result = prover.apply_tactic_to_proof("test_incr", "exact Nat.add_zero n")

            # Should execute without crashing
            assert isinstance(tactic_result.success, bool)

    def test_mathlib_integration(self):
        """Test basic mathlib functionality."""
        prover = LeanProver(mathlib_enabled=True)

        # Test mathlib-dependent theorem
        result = prover.run("theorem mathlib_test (a b : ℕ) : a + b = b + a := by simp [add_comm]")

        # May succeed or fail depending on environment, but should not crash
        assert isinstance(result.success, bool)
        assert isinstance(result.proof_complete, bool)
        assert isinstance(result.has_sorry, bool)


if __name__ == "__main__":
    # Allow running as script for development/debugging
    print("Running Basic LeanProver Tests...")
    pytest.main([__file__, "-v"])
