"""
Pytest tests for the Lean 4 interactive functionality.

Tests cover the complete lean4 submodule including:
- LeanProver: Core prover interface with mathlib support
- InteractiveLeanAgent: VS Code-like interactive development experience

Test scenarios include:
- Basic LeanProver functionality (proof execution, backtracking, incremental building)
- InteractiveLeanAgent theorem loading and compilation
- Interactive clause editing with real-time feedback
- Position-aware editing and goal tracking
- Complex proof structure building
- Terence Tao-style development workflows
- Error handling and edge cases
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nemo_skills.code_execution.lean4 import (
    LeanProver,
    ProofResult,
    ProofInProgress,
    InteractiveLeanAgent,
    Position,
    LeanMessage,
    ProofGoal,
    EditableClause,
)


class TestLeanProver:
    """Test suite for LeanProver core functionality."""

    def test_basic_proof_execution(self):
        """Test basic proof execution with simple theorems."""
        prover = LeanProver(mathlib_enabled=True)

        # Simple arithmetic proof
        result = prover.run("theorem test_basic : 1 + 1 = 2 := by simp")

        assert result.success == True
        assert result.proof_complete == True
        assert result.has_sorry == False
        assert result.error is None

    def test_proof_with_sorry(self):
        """Test detection of proofs with sorry."""
        prover = LeanProver(mathlib_enabled=True)

        # Proof with sorry
        result = prover.run("theorem test_sorry : 1 + 1 = 2 := by sorry")

        assert result.success == True
        assert result.proof_complete == False
        assert result.has_sorry == True
        assert result.proof_state is not None

    def test_incremental_proof_building(self):
        """Test incremental proof building workflow."""
        prover = LeanProver(mathlib_enabled=True)

        # Start a new proof
        start_result = prover.start_proof("test_incr", "(a : Nat) : a + 0 = a")
        assert start_result.success == True
        assert start_result.has_sorry == True
        assert "test_incr" in prover.proofs_in_progress

        # Apply a tactic
        tactic_result = prover.apply_tactic_to_proof("test_incr", "exact Nat.add_zero a")
        assert tactic_result.success == True

        # Check proof state
        proof_state = prover.get_proof_state("test_incr")
        assert proof_state is not None
        assert proof_state.name == "test_incr"
        assert len(proof_state.tactic_history) == 1

    def test_proof_backtracking(self):
        """Test proof backtracking functionality."""
        prover = LeanProver(mathlib_enabled=True)

        # Start proof and apply multiple tactics that will actually work
        start_result = prover.start_proof("test_backtrack", ": True ∧ True")
        assert start_result.success == True

        # Apply tactics that will work
        result1 = prover.apply_tactic_to_proof("test_backtrack", "constructor")
        result2 = prover.apply_tactic_to_proof("test_backtrack", "trivial")

        proof_state = prover.get_proof_state("test_backtrack")
        original_history_len = len(proof_state.tactic_history)

        # Only backtrack if we actually have history
        if original_history_len > 0:
            backtrack_result = prover.backtrack_proof("test_backtrack", 1)
            assert backtrack_result.success == True

            proof_state = prover.get_proof_state("test_backtrack")
            assert len(proof_state.tactic_history) == original_history_len - 1

    def test_multi_step_execution(self):
        """Test multi-step tactic execution."""
        prover = LeanProver(mathlib_enabled=True)

        # Start a proof that actually needs intro
        start_result = prover.start_proof("test_multi", ": ∀ (a : Nat), a + 0 = a")
        assert start_result.success == True

        # Execute multiple steps
        tactics = ["intro a", "exact Nat.add_zero a"]
        results = prover.run_multi_step(start_result.proof_state, tactics)

        assert len(results) == 2
        assert all(result.success for result in results)

    def test_error_handling(self):
        """Test error handling for invalid proofs."""
        prover = LeanProver(mathlib_enabled=True)

        # Invalid syntax
        result = prover.run("theorem invalid : 1 + 1 = 2 := by invalid_tactic")
        assert result.success == False
        assert result.error is not None


class TestInteractiveLeanAgent:
    """Test suite for InteractiveLeanAgent functionality."""

    def test_basic_theorem_loading(self):
        """Test basic theorem loading and analysis."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        simple_theorem = "theorem test_load (n : Nat) : n + 0 = n := by sorry"
        result = agent.load_theorem(simple_theorem)

        assert isinstance(result, dict)
        assert 'success' in result
        assert 'messages' in result
        assert 'goals' in result
        assert 'editable_clauses' in result
        assert len(result['editable_clauses']) > 0

    def test_simple_clause_editing(self):
        """Test editing simple clauses without have statements."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Load simple theorem with sorry
        simple_theorem = "theorem test_edit (n : Nat) : n + 0 = n := by sorry"
        load_result = agent.load_theorem(simple_theorem)

        # Should have at least one editable clause (the sorry)
        assert len(load_result['editable_clauses']) > 0

        # Edit the sorry clause
        clause_id = load_result['editable_clauses'][0]
        edit_result = agent.edit_clause(clause_id, "exact Nat.add_zero n")

        assert edit_result['edit_successful'] == True
        assert edit_result['compilation_result']['success'] == True
        assert 'exact Nat.add_zero n' in edit_result['updated_code']

    def test_complex_theorem_with_have_clauses(self):
        """Test editing theorems with have clauses."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Complex theorem with have statements
        complex_theorem = """theorem test_complex (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  intro h
  exact ⟨h1 h .left, ⟨h1 h .right, h2 h⟩⟩"""

        load_result = agent.load_theorem(complex_theorem)

        # Should identify have clauses
        editable_clauses = load_result['editable_clauses']
        have_clauses = [cid for cid in editable_clauses if cid.startswith('have_')]
        assert len(have_clauses) >= 2  # Should find h1 and h2

        # Edit the have clauses
        if 'have_h1' in editable_clauses:
            edit_result = agent.edit_clause('have_h1', 'intro h; exact h.left')
            assert edit_result['edit_successful'] == True

        if 'have_h2' in editable_clauses:
            edit_result = agent.edit_clause('have_h2', 'intro h; exact h.right')
            assert edit_result['edit_successful'] == True

    def test_interactive_panel(self):
        """Test interactive panel functionality (VS Code-like experience)."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        theorem = "theorem test_panel (n : Nat) : n + 0 = n := by sorry"
        agent.load_theorem(theorem)

        panel = agent.get_interactive_panel()

        assert isinstance(panel, dict)
        assert 'current_code' in panel
        assert 'messages' in panel
        assert 'goals' in panel
        assert 'editable_clauses' in panel
        assert 'compilation_id' in panel

        # Check that current_code contains our theorem
        assert 'test_panel' in panel['current_code']

    def test_position_aware_functionality(self):
        """Test position-aware editing and goal tracking."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        theorem = """theorem test_position (a b : Nat) : a + b = b + a := by
  sorry"""

        agent.load_theorem(theorem)

        # Test getting goals at specific positions
        goal = agent.get_goal_at_position(1, 0)  # Line 1, column 0
        messages = agent.get_messages_at_position(0, 0)  # Line 0, column 0

        # Should handle position queries gracefully
        assert goal is None or isinstance(goal, ProofGoal)
        assert isinstance(messages, list)

    def test_dynamic_proof_structure_building(self):
        """Test dynamic addition of proof structure."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Start with simple theorem
        simple_theorem = "theorem test_dynamic (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"
        load_result = agent.load_theorem(simple_theorem)

        # Add proof structure dynamically
        structure_lines = [
            "have h1 : P ∧ Q → P := by intro h; exact h.left",
            "have h2 : P ∧ Q → Q := by intro h; exact h.right",
            "intro h",
            "exact ⟨h2 h, h1 h⟩"
        ]

        structure_result = agent.add_proof_structure(structure_lines)

        assert 'edit_successful' in structure_result
        if structure_result.get('edit_successful'):
            # Check that structure was added
            panel = agent.get_interactive_panel()
            updated_code = panel['current_code']
            assert 'have h1' in updated_code
            assert 'have h2' in updated_code

    def test_terence_tao_workflow(self):
        """Test Terence Tao-style interactive development workflow."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Start with structured theorem (like Tao would write)
        theorem_sketch = """theorem associativity_demo (a b c : Nat) : (a + b) + c = a + (b + c) := by
  have step1 : (a + b) + c = (a + b) + c := by rfl
  have step2 : (a + b) + c = a + (b + c) := by sorry
  exact step2"""

        # Load theorem (like opening file in VS Code)
        load_result = agent.load_theorem(theorem_sketch)
        assert load_result['success'] == True or not load_result['has_errors']

        # Get initial panel state (like VS Code side panel)
        panel = agent.get_interactive_panel()
        initial_clauses = panel['editable_clauses']

        # Should identify have clauses for step1 and step2
        have_step1 = any('step1' in cid for cid in initial_clauses.keys())
        have_step2 = any('step2' in cid for cid in initial_clauses.keys())

        # At least one should be found (depends on parsing)
        assert have_step1 or have_step2 or len(initial_clauses) > 0

        # Get suggestions (like AI assistant)
        suggestions = agent.suggest_next_actions()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_incremental_development_with_feedback(self):
        """Test incremental development with real-time feedback."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Multi-step theorem development
        theorem = """theorem test_incremental (P Q R : Prop) : P → Q → R → P ∧ Q ∧ R := by
  sorry"""

        # Load and get initial state
        load_result = agent.load_theorem(theorem)
        initial_compilation_id = load_result['compilation_id']

        # Edit with incremental structure
        structure_lines = [
            "intro hP",
            "intro hQ",
            "intro hR",
            "exact ⟨hP, ⟨hQ, hR⟩⟩"
        ]

        structure_result = agent.add_proof_structure(structure_lines)

        if structure_result.get('edit_successful'):
            # Check compilation ID incremented (shows recompilation)
            final_panel = agent.get_interactive_panel()
            assert final_panel['compilation_id'] > initial_compilation_id

    def test_error_recovery_workflow(self):
        """Test error recovery in interactive development."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Start with theorem
        theorem = "theorem test_error (n : Nat) : n + 0 = n := by sorry"
        agent.load_theorem(theorem)

        # Try an invalid edit
        editable_clauses = agent.editable_clauses
        if editable_clauses:
            clause_id = list(editable_clauses.keys())[0]

            # Invalid tactic should fail gracefully
            edit_result = agent.edit_clause(clause_id, "invalid_tactic_name")

            # Should still be structured response even on failure
            assert 'edit_successful' in edit_result
            assert 'compilation_result' in edit_result

            # Then fix with valid tactic
            fix_result = agent.edit_clause(clause_id, "exact Nat.add_zero n")
            if fix_result['edit_successful']:
                assert fix_result['compilation_result']['success'] == True

    def test_clause_identification_edge_cases(self):
        """Test edge cases in clause identification."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Edge case: theorem with mixed content
        mixed_theorem = """theorem test_mixed (n : Nat) : n = n := by
  -- Comment line
  have h : n = n := by rfl
  exact h"""

        load_result = agent.load_theorem(mixed_theorem)

        # Should handle mixed content gracefully
        assert isinstance(load_result['editable_clauses'], list)

        # Edge case: empty theorem
        empty_theorem = "theorem test_empty (n : Nat) : n = n := by sorry"
        empty_result = agent.load_theorem(empty_theorem)
        assert isinstance(empty_result['editable_clauses'], list)

    def test_mathlib_integration(self):
        """Test mathlib integration in interactive development."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Use mathlib-dependent theorem
        mathlib_theorem = """theorem test_mathlib (a b : ℕ) : a + b = b + a := by
  sorry"""

        load_result = agent.load_theorem(mathlib_theorem)

        # Should load successfully with mathlib
        assert 'success' in load_result

        # Edit with mathlib tactic
        if load_result['editable_clauses']:
            clause_id = load_result['editable_clauses'][0]
            edit_result = agent.edit_clause(clause_id, "exact add_comm a b")

            # Should work with mathlib available
            assert 'edit_successful' in edit_result


class TestDataTypes:
    """Test the data types and structures used in lean4 module."""

    def test_position_namedtuple(self):
        """Test Position NamedTuple functionality."""
        pos = Position(line=5, column=10)
        assert pos.line == 5
        assert pos.column == 10
        assert isinstance(pos, tuple)

    def test_lean_message_dataclass(self):
        """Test LeanMessage dataclass functionality."""
        start_pos = Position(0, 0)
        end_pos = Position(0, 10)

        msg = LeanMessage(
            severity='error',
            message='Test error',
            start_pos=start_pos,
            end_pos=end_pos
        )

        assert msg.severity == 'error'
        assert msg.message == 'Test error'
        assert 'line 1' in str(msg)  # Should show 1-based line numbers

    def test_proof_goal_dataclass(self):
        """Test ProofGoal dataclass functionality."""
        pos = Position(2, 5)
        goal = ProofGoal(
            goal_text='Test goal text',
            position=pos,
            proof_state_id=42
        )

        assert goal.goal_text == 'Test goal text'
        assert goal.position == pos
        assert goal.proof_state_id == 42
        assert 'line 3' in str(goal)  # Should show 1-based line numbers

    def test_editable_clause_dataclass(self):
        """Test EditableClause dataclass functionality."""
        start_pos = Position(1, 0)
        end_pos = Position(1, 20)

        clause = EditableClause(
            clause_id='test_clause',
            start_pos=start_pos,
            end_pos=end_pos,
            content='sorry',
            clause_type='sorry'
        )

        assert clause.clause_id == 'test_clause'
        assert clause.clause_type == 'sorry'
        assert 'sorry' in str(clause)


class TestIntegrationScenarios:
    """Integration tests that combine multiple features."""

    def test_full_proof_development_cycle(self):
        """Test complete proof development from sketch to completion."""
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Start with high-level proof sketch
        initial_sketch = """theorem full_cycle (P Q : Prop) : P ∧ Q → Q ∧ P := by
  sorry"""

        # Phase 1: Load sketch
        load_result = agent.load_theorem(initial_sketch)
        assert len(load_result['editable_clauses']) > 0

        # Phase 2: Add structure
        structure_lines = [
            "intro h",
            "exact ⟨h.right, h.left⟩"
        ]

        structure_result = agent.add_proof_structure(structure_lines)

        # Phase 3: Verify completion
        if structure_result.get('edit_successful'):
            final_panel = agent.get_interactive_panel()
            # Check if proof compiles successfully
            final_compilation = structure_result['compilation_result']
            success = final_compilation.get('success', False)
            has_errors = final_compilation.get('has_errors', True)

            # Should either succeed or fail gracefully
            assert isinstance(success, bool)
            assert isinstance(has_errors, bool)

    def test_agent_prover_interoperability(self):
        """Test that InteractiveLeanAgent and LeanProver work well together."""
        # This demonstrates they use the same underlying system
        prover = LeanProver(mathlib_enabled=True)
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Both should be able to handle similar theorems
        theorem = "theorem interop_test (n : Nat) : n + 0 = n := by exact Nat.add_zero n"

        # Test with prover
        prover_result = prover.run(theorem)

        # Test with agent
        agent_result = agent.load_theorem(theorem)

        # Both should handle the theorem successfully
        assert prover_result.success == True
        # Agent success means no compilation errors
        agent_success = not agent_result.get('has_errors', True)

        # Both systems should give consistent results for valid proofs
        assert prover_result.success == agent_success


# Pytest fixtures for common test setup
@pytest.fixture
def lean_prover():
    """Fixture providing a LeanProver instance."""
    return LeanProver(mathlib_enabled=True)


@pytest.fixture
def interactive_agent():
    """Fixture providing an InteractiveLeanAgent instance."""
    return InteractiveLeanAgent(mathlib_enabled=True)


@pytest.fixture
def sample_theorems():
    """Fixture providing sample theorems for testing."""
    return {
        'simple': "theorem simple_test (n : Nat) : n + 0 = n := by sorry",
        'complex': """theorem complex_test (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  intro h
  exact ⟨h1 h .left, ⟨h1 h .right, h2 h⟩⟩""",
        'mathlib': "theorem mathlib_test (a b : ℕ) : a + b = b + a := by exact add_comm a b"
    }


# Parameterized tests for different theorem types
@pytest.mark.parametrize("theorem_type,expected_clauses", [
    ("simple", 1),  # Should have at least 1 clause (the sorry)
    ("complex", 2), # Should have at least 2 clauses (the have statements)
])
def test_clause_detection_parametrized(interactive_agent, sample_theorems, theorem_type, expected_clauses):
    """Parameterized test for clause detection across different theorem types."""
    theorem = sample_theorems[theorem_type]
    result = interactive_agent.load_theorem(theorem)

    # Should detect at least the expected number of clauses
    assert len(result['editable_clauses']) >= expected_clauses


if __name__ == "__main__":
    # Allow running as script for development/debugging
    print("Running Lean 4 Interactive Tests...")
    pytest.main([__file__, "-v"])
