#!/usr/bin/env python3
"""
Demo script showing how an LLM would use the advanced LeanProver capabilities.

This demonstrates:
- Standalone command execution
- Environment continuity across commands
- Building up definitions and theorems incrementally
- ProofStep operations (when applicable)
- Complex proof workflows that an LLM might follow
"""

from nemo_skills.code_execution.lean_prover import LeanProver, quick_command


def demo_basic_command_execution():
    """Demo basic command execution capabilities."""
    print("=== Basic Command Execution ===")

    with LeanProver() as prover:
        # Execute a simple theorem
        result = prover.run_command("theorem basic_thm : 1 + 1 = 2 := rfl")
        print(f"Basic theorem: success={result.success}, env={result.env}")

        # Check the theorem
        result = prover.run_command("#check basic_thm")
        print(f"Check theorem: success={result.success}")
        print(f"Response: {result.response_text[:100]}...")


def demo_environment_continuity():
    """Demo environment continuity - key for LLM interactions."""
    print("\n=== Environment Continuity ===")

    with LeanProver() as prover:
        print(f"Initial environment: {prover.get_current_env()}")

        # Step 1: Define a helper lemma
        result1 = prover.run_command("theorem helper_lemma : True ∨ False := Or.inl trivial")
        print(f"Helper lemma: success={result1.success}, env={result1.env}")

        # Step 2: Use the helper in a new theorem
        result2 = prover.run_command("theorem uses_helper : True ∨ False := helper_lemma")
        print(f"Uses helper: success={result2.success}, env={result2.env}")

        # Step 3: Check both theorems exist
        check1 = prover.run_command("#check helper_lemma")
        check2 = prover.run_command("#check uses_helper")
        print(f"Both checks successful: {check1.success and check2.success}")

        print(f"Final environment: {prover.get_current_env()}")


def demo_incremental_proof_building():
    """Demo building up complex proofs incrementally - LLM workflow."""
    print("\n=== Incremental Proof Building ===")

    with LeanProver() as prover:
        # Step 1: Start with basic definitions
        result = prover.run_command("def my_pred (x : Nat) : Prop := x > 0")
        print(f"Definition: success={result.success}")

        # Step 2: Prove a basic property
        result = prover.run_command("theorem basic_prop : my_pred 1 := by simp [my_pred]")
        if not result.success:
            # Fallback to simpler proof
            result = prover.run_command("theorem basic_prop : my_pred 1 := by unfold my_pred; norm_num")
        print(f"Basic property: success={result.success}")

        # Step 3: Build on previous results
        if result.success:
            result = prover.run_command("#check basic_prop")
            print(f"Can reference previous result: {result.success}")


def demo_proof_exploration_workflow():
    """Demo how an LLM might explore proof strategies."""
    print("\n=== Proof Exploration Workflow ===")

    with LeanProver() as prover:
        # LLM might try different approaches to the same theorem

        # Approach 1: Direct proof
        theorem_stmt = "theorem exploration_test : True → True"

        # Try simple approach first
        result1 = prover.run_command(f"{theorem_stmt} := id")
        print(f"Direct approach: success={result1.success}")

        if not result1.success:
            # Try tactic proof
            result2 = prover.run_command(f"{theorem_stmt} := by intro h; exact h")
            print(f"Tactic approach: success={result2.success}")

        # LLM could also try building with sorry first
        sorry_result = prover.run_command(f"theorem exploration_sorry : True → True := by intro h; sorry")
        print(f"Sorry structure: success={sorry_result.success}, complete={sorry_result.proof_complete}, has_sorry={sorry_result.has_sorry}")


def demo_error_recovery_workflow():
    """Demo how an LLM might recover from errors."""
    print("\n=== Error Recovery Workflow ===")

    with LeanProver() as prover:
        # Try an approach that might fail
        result = prover.run_command("theorem recovery_test : 1 + 1 = 2 := by unknown_tactic")
        print(f"Failed attempt: success={result.success}")
        print(f"Error: {result.error_message[:100]}...")

        # LLM recovers by trying a different approach
        if not result.success:
            result2 = prover.run_command("theorem recovery_test : 1 + 1 = 2 := rfl")
            print(f"Recovery attempt: success={result2.success}")


def demo_proof_state_operations():
    """Demo ProofStep operations (when proof states are available)."""
    print("\n=== Proof State Operations ===")

    with LeanProver() as prover:
        # These operations require valid proof states from the server
        # In a real LLM workflow, these would come from previous operations

        print("Testing proof state interface...")

        # Test single tactic application
        result = prover.run_proof_step_on_state(0, "intro h")
        print(f"Single tactic on state 0: success={result.success}")
        if not result.success:
            print(f"Expected - state 0 may not exist: {result.error_message[:100]}...")

        # Test multi-tactic application
        multi_tactic = "(\nintro h\nexact h)"
        result = prover.run_multi_step_on_state(0, multi_tactic)
        print(f"Multi-tactic on state 0: success={result.success}")

        print("Note: ProofStep operations need valid proof states from ongoing proofs")


def demo_complex_llm_workflow():
    """Demo a complex workflow an LLM might follow."""
    print("\n=== Complex LLM Workflow ===")

    with LeanProver() as prover:
        print("LLM following a complex proof strategy...")

        # Phase 1: Explore the problem
        exploration = prover.run_command("#check And")
        print(f"Phase 1 - Explore types: success={exploration.success}")

        # Phase 2: Try a structured approach with sorry
        structure = prover.run_command("""
theorem complex_example (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  constructor
  · sorry  -- Will prove Q from P ∧ Q
  · sorry  -- Will prove P from P ∧ Q
""")
        print(f"Phase 2 - Structure with sorry: success={structure.success}, has_sorry={structure.has_sorry}")

        # Phase 3: Fill in the details
        if structure.success:
            complete = prover.run_command("""
theorem complex_complete (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  constructor
  · exact h.2  -- Get Q from h : P ∧ Q
  · exact h.1  -- Get P from h : P ∧ Q
""")
            print(f"Phase 3 - Complete proof: success={complete.success}, complete={complete.proof_complete}")


def demo_convenience_functions():
    """Demo convenience functions for quick operations."""
    print("\n=== Convenience Functions ===")

    # Quick command execution
    result = quick_command("theorem quick_test : True := trivial")
    print(f"Quick command: success={result.success}, complete={result.proof_complete}")

    # Show environment was automatically managed
    result2 = quick_command("#check quick_test")
    print(f"Quick check: success={result2.success}")


def main():
    """Run all demonstrations."""
    print("LLM-Oriented Lean Prover Capabilities Demo")
    print("==========================================")

    try:
        demo_basic_command_execution()
        demo_environment_continuity()
        demo_incremental_proof_building()
        demo_proof_exploration_workflow()
        demo_error_recovery_workflow()
        demo_proof_state_operations()
        demo_complex_llm_workflow()
        demo_convenience_functions()

        print("\n=== Summary of LLM-Relevant Features ===")
        print("✅ Standalone command execution")
        print("✅ Environment continuity across operations")
        print("✅ Incremental definition building")
        print("✅ Error recovery workflows")
        print("✅ Sorry-based proof structuring")
        print("✅ Proof state manipulation interface")
        print("✅ Quick convenience functions")
        print("\n🎯 These features enable sophisticated LLM proof interactions!")

    except Exception as e:
        print(f"Demo error: {e}")
        print("This is normal - some advanced features depend on server state")


if __name__ == "__main__":
    main()
