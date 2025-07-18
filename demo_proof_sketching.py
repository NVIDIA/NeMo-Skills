#!/usr/bin/env python3
"""
Demo of Proof Sketching with lean-interact

This demonstrates how to break down complex theorems into smaller pieces
using `have` statements with `sorry`, then work on each piece incrementally.
This is perfect for LLM-driven theorem proving workflows.
"""

from nemo_skills.code_execution.lean_prover import LeanProver

def demo_proof_sketching():
    """Demonstrate proof sketching by breaking down a theorem."""
    print("=== Proof Sketching Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # First, let's sketch a more complex theorem by breaking it down
    print("1. Sketching a complex theorem with 'have' statements...")

    complex_theorem = """
theorem complex_example (a b c : ℕ) (h : a + b = c) : a + (b + 1) = c + 1 := by
  have h1 : a + b + 1 = c + 1 := by sorry
  have h2 : a + (b + 1) = a + b + 1 := by sorry
  rw [h2, h1]
"""

    result = prover.run_command(complex_theorem)
    print(f"   Sketch created: {result.proof_state is not None}")
    print(f"   Number of sorries: {len(result.response.split('sorry'))}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        # Now we can work on each sorry separately
        print("\n2. Working on individual subgoals...")

        # Get the proof state and work on the first sorry
        print("   Working on h1: a + b + 1 = c + 1...")

        # We'll need to identify which proof state corresponds to which sorry
        # Let's apply a tactic to see what happens
        step1 = prover.run_proof_step(result.proof_state, "simp [Nat.add_assoc]")
        print(f"      Applied simp: success={step1.success}")
        if step1.goals:
            print(f"      Remaining goals: {len(step1.goals)}")
            for i, goal in enumerate(step1.goals):
                print(f"         Goal {i+1}: {goal.split('⊢')[1].strip() if '⊢' in goal else goal}")

def demo_structured_proof_building():
    """Demonstrate building a structured proof step by step."""
    print("\n=== Structured Proof Building Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # Let's use a theorem that naturally breaks down into parts
    print("1. Creating a theorem that proves commutativity through intermediate steps...")

    structured_theorem = """
theorem comm_proof (a b : ℕ) : a + b = b + a := by
  have h1 : a + b = a + b := by sorry
  have h2 : a + b = b + a := by sorry
  exact h2
"""

    result = prover.run_command(structured_theorem)
    print(f"   Structured proof created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        # Work on the first sorry (trivial case)
        print("\n2. Working on h1 (trivial reflexivity)...")
        step1 = prover.run_proof_step(result.proof_state, "rfl")
        print(f"      Applied rfl: success={step1.success}")
        if step1.goals:
            print(f"      Remaining goals: {len(step1.goals)}")

def demo_mathematical_breakdown():
    """Demonstrate breaking down a mathematical proof into logical steps."""
    print("\n=== Mathematical Breakdown Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # A more interesting mathematical example
    print("1. Breaking down a proof about natural number arithmetic...")

    math_theorem = """
theorem nat_identity (n : ℕ) : n + 0 = n ∧ 0 + n = n := by
  have left_identity : n + 0 = n := by sorry
  have right_identity : 0 + n = n := by sorry
  exact ⟨left_identity, right_identity⟩
"""

    result = prover.run_command(math_theorem)
    print(f"   Mathematical proof sketched: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        # Let's work on each identity
        print("\n2. Solving the left identity: n + 0 = n...")
        step1 = prover.run_proof_step(result.proof_state, "simp")
        print(f"      Applied simp: success={step1.success}")
        if step1.goals:
            print(f"      Remaining goals: {len(step1.goals)}")
            for i, goal in enumerate(step1.goals):
                goal_text = goal.split('⊢')[1].strip() if '⊢' in goal else goal
                print(f"         Goal {i+1}: {goal_text}")

def demo_llm_style_proof_workflow():
    """Demonstrate the kind of workflow an LLM would use."""
    print("\n=== LLM-Style Proof Workflow Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # This is how an LLM might approach a proof:
    # 1. Identify the main structure
    # 2. Break it down into manageable pieces
    # 3. Work on each piece
    # 4. Combine the results

    print("1. LLM identifies proof structure and creates skeleton...")

    llm_style_proof = """
theorem distributivity (a b c : ℕ) : a * (b + c) = a * b + a * c := by
  -- LLM identifies key steps:
  have step1 : a * (b + c) = a * (b + c) := by sorry  -- trivial
  have step2 : a * (b + c) = a * b + a * c := by sorry  -- main work
  exact step2
"""

    result = prover.run_command(llm_style_proof)
    print(f"   LLM-style proof skeleton created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        print("\n2. LLM works on step1 (trivial case)...")
        step1 = prover.run_proof_step(result.proof_state, "rfl")
        print(f"      Applied rfl: success={step1.success}")

        print("\n3. LLM works on step2 (main distributivity)...")
        # This would be where the LLM applies the distributivity rule
        step2 = prover.run_proof_step(step1.proof_state if step1.success else result.proof_state, "simp [Nat.mul_add]")
        print(f"      Applied simp [Nat.mul_add]: success={step2.success}")

        if step2.goals:
            print(f"      Remaining goals: {len(step2.goals)}")
            if len(step2.goals) == 0:
                print("      ✅ Proof completed!")
            else:
                for i, goal in enumerate(step2.goals):
                    goal_text = goal.split('⊢')[1].strip() if '⊢' in goal else goal
                    print(f"         Goal {i+1}: {goal_text}")

if __name__ == "__main__":
    demo_proof_sketching()
    demo_structured_proof_building()
    demo_mathematical_breakdown()
    demo_llm_style_proof_workflow()

    print("\n" + "="*60)
    print("🎯 PROOF SKETCHING WORKFLOW SUMMARY:")
    print("   ✅ Break complex theorems into 'have' statements with 'sorry'")
    print("   ✅ Each 'sorry' creates a proof state that can be worked on")
    print("   ✅ Work on each subgoal independently using ProofStep")
    print("   ✅ Perfect for LLM-driven theorem proving!")
    print("   ✅ Enables incremental proof construction and refinement")
    print("="*60)
