#!/usr/bin/env python3
"""
Demo of Breaking Down Theorems into Multiple 'have' Statements

This shows the exact workflow the user requested:
1. Start with a complex theorem
2. Break it down into 'have h1...by sorry', 'have h2...by sorry', etc.
3. Work on each sorry independently using proof states
"""

from nemo_skills.code_execution.lean_prover import LeanProver

def demo_theorem_breakdown():
    """Demonstrate breaking down a theorem into multiple have statements."""
    print("=== Theorem Breakdown Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # Example 1: Simple arithmetic proof broken down
    print("1. Creating a theorem broken down into steps...")

    # Instead of one complex proof, we break it into logical steps
    broken_down_proof = """
theorem arithmetic_breakdown (a b : ℕ) : (a + b) + 1 = a + (b + 1) := by
  have h1 : (a + b) + 1 = a + (b + 1) := by sorry
  exact h1
"""

    result = prover.run_command(broken_down_proof)
    print(f"   Theorem created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Proof state for first sorry: {result.proof_state}")

        # Work on the sorry
        print("\n2. Working on h1 using proof state...")
        step1 = prover.run_proof_step(result.proof_state, "rw [Nat.add_assoc]")
        print(f"      Applied rw [Nat.add_assoc]: success={step1.success}")
        print(f"      Proof complete: {step1.proof_complete}")

def demo_multiple_have_statements():
    """Demonstrate multiple have statements with sorry."""
    print("\n=== Multiple Have Statements Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # Example with multiple intermediate steps
    print("1. Creating theorem with multiple 'have' statements...")

    multi_have_proof = """
theorem multi_step_proof (a b c : ℕ) : a + b + c = c + b + a := by
  have step1 : a + b + c = a + (b + c) := by sorry
  have step2 : a + (b + c) = a + (c + b) := by sorry
  have step3 : a + (c + b) = (a + c) + b := by sorry
  have step4 : (a + c) + b = (c + a) + b := by sorry
  have step5 : (c + a) + b = c + (a + b) := by sorry
  have step6 : c + (a + b) = c + (b + a) := by sorry
  have step7 : c + (b + a) = c + b + a := by sorry
  rw [step1, step2, step3, step4, step5, step6, step7]
"""

    result = prover.run_command(multi_have_proof)
    print(f"   Multi-step theorem created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        # Try working on the first sorry
        print("\n2. Working on step1: a + b + c = a + (b + c)...")
        step1 = prover.run_proof_step(result.proof_state, "rw [Nat.add_assoc]")
        print(f"      Applied rw [Nat.add_assoc]: success={step1.success}")
        if step1.success:
            print(f"      New proof state: {step1.proof_state}")

            # Try working on the next sorry
            print("\n3. Working on step2: a + (b + c) = a + (c + b)...")
            step2 = prover.run_proof_step(step1.proof_state, "rw [Nat.add_comm b c]")
            print(f"      Applied rw [Nat.add_comm b c]: success={step2.success}")
            if step2.success:
                print(f"      New proof state: {step2.proof_state}")

def demo_realistic_proof_sketch():
    """Demonstrate a realistic proof sketching workflow."""
    print("\n=== Realistic Proof Sketch Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # A more realistic example that an LLM might create
    print("1. LLM sketches a proof about list operations...")

    realistic_sketch = """
theorem list_operations (l : List ℕ) : l.length = l.reverse.length := by
  have base_case : ([] : List ℕ).length = ([] : List ℕ).reverse.length := by sorry
  have inductive_step : ∀ (h : ℕ) (t : List ℕ),
    t.length = t.reverse.length →
    (h :: t).length = (h :: t).reverse.length := by sorry
  -- Would normally use induction, but for sketch:
  sorry
"""

    result = prover.run_command(realistic_sketch)
    print(f"   Realistic sketch created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        # Work on the base case
        print("\n2. Working on base case...")
        step1 = prover.run_proof_step(result.proof_state, "simp [List.length, List.reverse]")
        print(f"      Applied simp: success={step1.success}")

def demo_incremental_proof_construction():
    """Show how to incrementally construct a proof by filling in sorries."""
    print("\n=== Incremental Proof Construction Demo ===\n")

    prover = LeanProver(mathlib_enabled=True)

    # Start with a basic sketch
    print("1. Starting with basic proof sketch...")

    basic_sketch = """
theorem incremental_example (x y : ℕ) : x + y = y + x := by
  have symmetry : x + y = y + x := by sorry
  exact symmetry
"""

    result = prover.run_command(basic_sketch)
    print(f"   Basic sketch created: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Proof state: {result.proof_state}")

        # Now fill in the sorry with the actual proof
        print("\n2. Filling in the sorry with actual proof...")
        step1 = prover.run_proof_step(result.proof_state, "rw [Nat.add_comm]")
        print(f"      Applied rw [Nat.add_comm]: success={step1.success}")
        print(f"      Proof complete: {step1.proof_complete}")

        if step1.proof_complete:
            print("      ✅ Incremental proof construction successful!")

def demo_llm_workflow_example():
    """Show exactly how an LLM would use this workflow."""
    print("\n=== LLM Workflow Example ===\n")

    prover = LeanProver(mathlib_enabled=True)

    print("🤖 LLM workflow:")
    print("   1. Analyze the theorem")
    print("   2. Identify key proof steps")
    print("   3. Create skeleton with 'have' statements")
    print("   4. Work on each step incrementally")
    print()

    llm_workflow = """
theorem llm_example (a b : ℕ) : a * (b + 1) = a * b + a := by
  -- LLM identifies: need to use distributivity
  have distributive : a * (b + 1) = a * b + a * 1 := by sorry
  have simplify : a * 1 = a := by sorry
  rw [distributive, simplify]
"""

    result = prover.run_command(llm_workflow)
    print(f"🤖 LLM created proof skeleton: {result.proof_state is not None}")

    if result.proof_state is not None:
        print(f"   Initial proof state: {result.proof_state}")

        print("\n🤖 LLM works on distributive step...")
        step1 = prover.run_proof_step(result.proof_state, "rw [Nat.mul_add]")
        print(f"      Applied rw [Nat.mul_add]: success={step1.success}")

        if step1.success:
            print("\n🤖 LLM works on simplification step...")
            step2 = prover.run_proof_step(step1.proof_state, "simp [Nat.mul_one]")
            print(f"      Applied simp [Nat.mul_one]: success={step2.success}")
            print(f"      Proof complete: {step2.proof_complete}")

if __name__ == "__main__":
    demo_theorem_breakdown()
    demo_multiple_have_statements()
    demo_realistic_proof_sketch()
    demo_incremental_proof_construction()
    demo_llm_workflow_example()

    print("\n" + "="*70)
    print("🎯 PROOF SKETCHING WORKFLOW SUMMARY:")
    print("✅ Break theorems into: theorem ... := by")
    print("     have h1 : ... := by sorry")
    print("     have h2 : ... := by sorry")
    print("     -- use h1, h2 to complete proof")
    print()
    print("✅ Each 'sorry' creates a proof state that can be worked on")
    print("✅ Work on each subgoal independently using ProofStep")
    print("✅ Perfect for LLM-driven incremental theorem proving!")
    print("✅ Enables structured proof development and refinement")
    print("="*70)
