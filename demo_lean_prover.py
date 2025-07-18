#!/usr/bin/env python3
"""
Demo script showing how to use the simple LeanProver interface.

This demonstrates:
- Basic proof steps
- The crucial distinction between sorry and complete proofs
- Partial proof progress with backtracking
- Mathlib integration
- Interactive proof development workflow
"""

from nemo_skills.code_execution.lean_prover import LeanProver, quick_prove


def demo_basic_proofs():
    """Demo basic proof functionality."""
    print("=== Basic Proof Demo ===")

    with LeanProver() as prover:
        # Simple complete proof
        result = prover.run_proof_step("theorem simple : 1 + 1 = 2", "norm_num")
        print(f"Simple proof: success={result.success}, complete={result.proof_complete}, has_sorry={result.has_sorry}")

        # Invalid tactic - should fail
        result = prover.run_proof_step("theorem simple : 1 + 1 = 2", "invalid_tactic")
        print(f"Invalid tactic: success={result.success}, complete={result.proof_complete}")
        print(f"Error: {result.error_message}")


def demo_sorry_vs_complete():
    """Demo the crucial distinction between sorry and complete proofs."""
    print("\n=== Sorry vs Complete Proof Demo ===")

    with LeanProver() as prover:
        # Complete proof
        complete = prover.run_proof_step("theorem test1 : 1 + 1 = 2", "norm_num")
        print(f"Complete proof: success={complete.success}, complete={complete.proof_complete}, has_sorry={complete.has_sorry}")

        # Proof with sorry - structure is valid but not complete
        sorry = prover.check_proof_with_sorry("theorem test2 : 1 + 1 = 2", "by sorry")
        print(f"Sorry proof: success={sorry.success}, complete={sorry.proof_complete}, has_sorry={sorry.has_sorry}")

        print(f"\nKEY INSIGHT: Both proofs have success=True (valid structure)")
        print(f"But only complete proof has proof_complete=True")
        print(f"The has_sorry flag clearly distinguishes them!")


def demo_partial_progress():
    """Demo partial progress and backtracking."""
    print("\n=== Partial Progress & Backtracking Demo ===")

    with LeanProver() as prover:
        theorem = "theorem test : (1 = 1) ∧ (2 = 2) ∧ (3 = 3)"

        # First attempt - partial progress
        result = prover.run_proof_step(theorem, "constructor")
        print(f"After constructor: success={result.success}, complete={result.proof_complete}, goals={len(result.goals)}")
        print(f"Remaining goals: {result.goals}")

        # User could backtrack and try a complete sequence
        result = prover.run_tactic_sequence(theorem, [
            "constructor", "norm_num",
            "constructor", "norm_num", "norm_num"
        ])
        print(f"Complete sequence: success={result.success}, complete={result.proof_complete}, goals={len(result.goals)}")


def demo_mathlib_integration():
    """Demo mathlib functionality."""
    print("\n=== Mathlib Integration Demo ===")

    with LeanProver(use_mathlib=True) as prover:
        # Natural number proof
        result = prover.run_proof_step("theorem nat_test : (0 : ℕ) + 1 = 1", "norm_num")
        print(f"Natural number proof: success={result.success}, complete={result.proof_complete}")

        # Real number proof
        result = prover.run_proof_step("theorem real_test : (0 : ℝ) + 1 = 1", "norm_num")
        print(f"Real number proof: success={result.success}, complete={result.proof_complete}")

        # More complex mathlib proof
        result = prover.run_proof_step("theorem simp_test : ∀ n : ℕ, n + 0 = n", "intro n; simp")
        print(f"Simp proof: success={result.success}, complete={result.proof_complete}")


def demo_proof_with_sorry_hypothesis():
    """Demo using sorry for hypotheses that could be proven later."""
    print("\n=== Sorry Hypothesis Demo ===")

    with LeanProver() as prover:
        # Create a proof outline with sorry for parts we'll implement later
        proof_outline = """by
  intro n
  -- We'll prove this lemma later
  have h : n + 0 = n := by sorry
  exact h"""

        result = prover.check_proof_with_sorry("theorem test : ∀ n : ℕ, n + 0 = n", proof_outline)
        print(f"Proof with sorry hypothesis: success={result.success}, complete={result.proof_complete}, has_sorry={result.has_sorry}")
        print("This allows you to structure proofs and fill in details later!")


def demo_interactive_workflow():
    """Demo an interactive proof development workflow."""
    print("\n=== Interactive Workflow Demo ===")

    with LeanProver() as prover:
        theorem = "theorem workflow : (A ∧ B) → (B ∧ A)"

        # Step 1: Start with the structure
        print("Step 1: Introduce the hypothesis")
        result = prover.run_proof_step(theorem, "intro h")
        print(f"After intro: success={result.success}, complete={result.proof_complete}, goals={len(result.goals)}")

        # Step 2: Try to make progress
        print("\nStep 2: Split the conjunction")
        result = prover.run_tactic_sequence(theorem, ["intro h", "constructor"])
        print(f"After constructor: success={result.success}, complete={result.proof_complete}, goals={len(result.goals)}")

        # Step 3: Complete the proof
        print("\nStep 3: Complete both goals")
        result = prover.run_tactic_sequence(theorem, [
            "intro h", "constructor",
            "exact h.2", "exact h.1"
        ])
        print(f"Final result: success={result.success}, complete={result.proof_complete}, goals={len(result.goals)}")


def demo_convenience_functions():
    """Demo the convenience functions for quick proofs."""
    print("\n=== Convenience Functions Demo ===")

    # Quick single proof
    result = quick_prove("theorem quick : 1 + 1 = 2", "norm_num")
    print(f"Quick prove: success={result.success}, complete={result.proof_complete}")

    # Quick sequence
    from nemo_skills.code_execution.lean_prover import quick_prove_sequence
    result = quick_prove_sequence(
        "theorem quick_seq : (1 = 1) ∧ (2 = 2)",
        ["constructor", "norm_num", "norm_num"]
    )
    print(f"Quick sequence: success={result.success}, complete={result.proof_complete}")


if __name__ == "__main__":
    print("LeanProver Simple Interface Demo")
    print("================================")

    demo_basic_proofs()
    demo_sorry_vs_complete()
    demo_partial_progress()
    demo_mathlib_integration()
    demo_proof_with_sorry_hypothesis()
    demo_interactive_workflow()
    demo_convenience_functions()

    print("\n=== Summary ===")
    print("Key features of the simple interface:")
    print("- Clear distinction between sorry and complete proofs")
    print("- User manages proof state - you get goals back for partial progress")
    print("- Easy backtracking by trying different tactic sequences")
    print("- Mathlib integration with TempRequireProject")
    print("- Simple server.run-based approach")
    print("- Context manager for automatic cleanup")
