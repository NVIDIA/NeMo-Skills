#!/usr/bin/env python3
"""
Demo of incremental proof building with tactic manipulation and backtracking.

This demonstrates:
- Starting a proof and building it step by step
- Adding tactics one at a time
- Inspecting intermediate proof states
- Removing tactics (backtracking)
- Branching to try different approaches
- Complete proof state management
"""

from nemo_skills.code_execution.lean_prover import LeanProver


def demo_incremental_proof_building():
    """Demo building a proof step by step."""
    print("=== Incremental Proof Building ===")

    with LeanProver(use_mathlib=False) as prover:
        # Start a proof
        theorem = "theorem demo : (A ∧ B) → (B ∧ A)"
        proof = prover.start_proof(theorem)

        print(f"Started proof: {theorem}")
        print(f"Initial state: goals={proof.goals}, complete={proof.is_complete}")
        print(f"Current proof text: {proof.get_current_proof_text()}")
        print()

        # Add first tactic
        proof = proof.apply_tactic(prover, "intro h")
        print(f"After 'intro h': goals={proof.goals}, complete={proof.is_complete}")
        print(f"Current proof text: {proof.get_current_proof_text()}")
        print()

        # Add second tactic
        proof = proof.apply_tactic(prover, "constructor")
        print(f"After 'constructor': goals={proof.goals}, complete={proof.is_complete}")
        print(f"Current proof text: {proof.get_current_proof_text()}")
        print()

        # Add third tactic
        proof = proof.apply_tactic(prover, "exact h.2")
        print(f"After 'exact h.2': goals={proof.goals}, complete={proof.is_complete}")
        print(f"Current proof text: {proof.get_current_proof_text()}")
        print()

        # Complete the proof
        proof = proof.apply_tactic(prover, "exact h.1")
        print(f"After 'exact h.1': goals={proof.goals}, complete={proof.is_complete}")
        print(f"Final proof text: {proof.get_current_proof_text()}")


def demo_tactic_manipulation():
    """Demo adding and removing tactics."""
    print("\n=== Tactic Manipulation ===")

    with LeanProver(use_mathlib=False) as prover:
        # Start a proof
        theorem = "theorem manipulation : True ∧ True"
        proof = prover.start_proof(theorem)

        print(f"Started: {theorem}")
        print(f"Tactics: {proof.tactics}")
        print()

        # Add some tactics
        proof = proof.apply_tactic(prover, "constructor")
        print(f"Added 'constructor': {proof.tactics}")

        proof = proof.apply_tactic(prover, "trivial")
        print(f"Added 'trivial': {proof.tactics}")

        proof = proof.apply_tactic(prover, "sorry")  # Wrong tactic
        print(f"Added 'sorry': {proof.tactics}")
        print()

        # Remove the last tactic
        proof = proof.remove_last_tactic()
        print(f"Removed last tactic: {proof.tactics}")

        # Try a different approach
        proof = proof.apply_tactic(prover, "trivial")
        print(f"Added 'trivial': {proof.tactics}")
        print(f"Final: complete={proof.is_complete}")


def demo_backtracking():
    """Demo backtracking to earlier proof states."""
    print("\n=== Backtracking Demo ===")

    with LeanProver(use_mathlib=False) as prover:
        # Start a proof
        theorem = "theorem backtrack : (A ∧ B) ∧ C → C ∧ (A ∧ B)"
        proof = prover.start_proof(theorem)

        print(f"Started: {theorem}")

        # Build up a proof with multiple steps
        proof = proof.apply_tactic(prover, "intro h")
        print(f"Step 1 - intro h: {proof.tactics}")

        proof = proof.apply_tactic(prover, "constructor")
        print(f"Step 2 - constructor: {proof.tactics}")

        proof = proof.apply_tactic(prover, "exact h.2")
        print(f"Step 3 - exact h.2: {proof.tactics}")

        proof = proof.apply_tactic(prover, "wrong_tactic")  # This might fail
        print(f"Step 4 - wrong_tactic: {proof.tactics}")
        print()

        # Backtrack to step 2
        proof = proof.backtrack_to(2)  # Back to after constructor
        print(f"Backtracked to step 2: {proof.tactics}")

        # Try a different approach from step 2
        proof = proof.apply_tactic(prover, "cases h")
        print(f"New approach - cases h: {proof.tactics}")


def demo_branching_exploration():
    """Demo exploring different proof strategies."""
    print("\n=== Branching Exploration ===")

    with LeanProver(use_mathlib=False) as prover:
        # Start a proof
        theorem = "theorem explore : True ∧ True"
        base_proof = prover.start_proof(theorem)

        print(f"Started: {theorem}")
        print("Exploring different approaches...")
        print()

        # Approach 1: Direct constructor
        approach1 = base_proof.apply_tactic(prover, "constructor")
        approach1 = approach1.apply_tactic(prover, "trivial")
        approach1 = approach1.apply_tactic(prover, "trivial")
        print(f"Approach 1 - constructor then trivial: complete={approach1.is_complete}")
        print(f"Tactics: {approach1.tactics}")
        print()

        # Approach 2: Use exact
        approach2 = base_proof.apply_tactic(prover, "exact ⟨trivial, trivial⟩")
        print(f"Approach 2 - exact: complete={approach2.is_complete}")
        print(f"Tactics: {approach2.tactics}")
        print()

        # Approach 3: Use split (if available)
        approach3 = base_proof.apply_tactic(prover, "constructor")
        approach3 = approach3.apply_tactic(prover, "apply True.intro")
        approach3 = approach3.apply_tactic(prover, "apply True.intro")
        print(f"Approach 3 - explicit True.intro: complete={approach3.is_complete}")
        print(f"Tactics: {approach3.tactics}")


def demo_proof_inspection():
    """Demo inspecting proof states at different stages."""
    print("\n=== Proof State Inspection ===")

    with LeanProver(use_mathlib=False) as prover:
        # Start a complex proof
        theorem = "theorem inspect : (A ∧ B) ∧ (C ∧ D) → (A ∧ C) ∧ (B ∧ D)"
        proof = prover.start_proof(theorem)

        print(f"Started: {theorem}")

        # Inspect initial state
        state = prover.inspect_proof_state(proof)
        print(f"Initial state: goals={state.goals}")
        print()

        # Add tactics and inspect at each stage
        proof = proof.apply_tactic(prover, "intro h")
        state = prover.inspect_proof_state(proof)
        print(f"After intro h: goals={state.goals}")

        proof = proof.apply_tactic(prover, "constructor")
        state = prover.inspect_proof_state(proof)
        print(f"After constructor: goals={state.goals}")

        proof = proof.apply_tactic(prover, "constructor")
        state = prover.inspect_proof_state(proof)
        print(f"After second constructor: goals={state.goals}")

        print(f"Current proof text:\n{proof.get_current_proof_text()}")


def main():
    """Run all demonstrations."""
    print("Incremental Proof Building Demo")
    print("===============================")

    try:
        demo_incremental_proof_building()
        demo_tactic_manipulation()
        demo_backtracking()
        demo_branching_exploration()
        demo_proof_inspection()

        print("\n=== Summary of New Capabilities ===")
        print("✅ Start proofs and build incrementally")
        print("✅ Add tactics one at a time")
        print("✅ Remove tactics (backtracking)")
        print("✅ Backtrack to specific steps")
        print("✅ Branch and explore different approaches")
        print("✅ Inspect proof states at any point")
        print("✅ Get current proof text at any stage")
        print("✅ Complete control over proof construction")
        print("\n🎯 Now you can fully manipulate proofs step by step!")

    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
