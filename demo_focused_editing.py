#!/usr/bin/env python3
"""
Focused Demo: Effective Sketch Editing with Lean-Interact

This demonstrates the KEY INSIGHT for sketch editing with lean-interact:
- You don't "edit" existing theorems
- You CREATE new versions of theorems
- Each version gets new proof states to work with
- This enables powerful iterative proof development
"""

from nemo_skills.code_execution.lean_prover import LeanProver

def demonstrate_editing_pattern():
    """Show the core editing pattern that works with lean-interact."""
    print("=" * 70)
    print("CORE EDITING PATTERN WITH LEAN-INTERACT")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n🎯 KEY INSIGHT: Don't edit - CREATE new versions!")
    print("Each new version gets fresh proof states to work with.")
    print()

    # Base version
    print("📄 BASE VERSION:")
    print("-" * 40)

    base_code = """theorem demo_base (P Q : Prop) : P ∧ Q → Q ∧ P := by
  sorry"""

    print("CODE:", base_code)
    base_result = prover.run_command(base_code)
    print(f"PROOF STATE: {base_result.proof_state}")
    print(f"CAN WORK ON: {'Yes' if base_result.proof_state is not None else 'No'}")

    # Enhanced version 1
    print("\n📄 ENHANCED VERSION 1 (add structure):")
    print("-" * 40)

    v1_code = """theorem demo_v1 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  have hp : P := by sorry
  have hq : Q := by sorry
  exact ⟨hq, hp⟩"""

    print("CODE:", v1_code)
    v1_result = prover.run_command(v1_code)
    print(f"PROOF STATE: {v1_result.proof_state}")
    print(f"CAN WORK ON: {'Yes' if v1_result.proof_state is not None else 'No'}")

    # Work on version 1
    if v1_result.proof_state is not None:
        print("\n🔧 WORKING ON VERSION 1:")
        print("   Filling hp: P...")
        step1 = prover.run_proof_step(v1_result.proof_state, "exact h.1")
        print(f"   Success: {step1.success}")

        if step1.success:
            print("   Filling hq: Q...")
            step2 = prover.run_proof_step(step1.proof_state, "exact h.2")
            print(f"   Success: {step2.success}")
            print(f"   Proof complete: {step2.proof_complete}")

    # Enhanced version 2 (different approach)
    print("\n📄 ENHANCED VERSION 2 (different approach):")
    print("-" * 40)

    v2_code = """theorem demo_v2 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  constructor
  · exact h.right
  · exact h.left"""

    print("CODE:", v2_code)
    v2_result = prover.run_command(v2_code)
    print(f"PROOF STATE: {v2_result.proof_state}")
    print(f"SUCCESS: {v2_result.success}")

    if v2_result.success:
        print("   🎉 This version works completely!")

    # Enhanced version 3 (one-liner)
    print("\n📄 ENHANCED VERSION 3 (simplified):")
    print("-" * 40)

    v3_code = """theorem demo_v3 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  exact ⟨h.2, h.1⟩"""

    print("CODE:", v3_code)
    v3_result = prover.run_command(v3_code)
    print(f"PROOF STATE: {v3_result.proof_state}")
    print(f"SUCCESS: {v3_result.success}")

    if v3_result.success:
        print("   🎉 This simplified version also works!")

def demonstrate_complex_editing():
    """Show editing with a more complex proof."""
    print("\n" + "=" * 70)
    print("COMPLEX PROOF EDITING EXAMPLE")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n🎯 Goal: Prove distributivity (P ∧ Q) ∨ (P ∧ R) ↔ P ∧ (Q ∨ R)")
    print()

    # Version 1: Just the structure
    print("📄 VERSION 1: Basic structure")
    v1 = """theorem distrib_v1 (P Q R : Prop) : (P ∧ Q) ∨ (P ∧ R) ↔ P ∧ (Q ∨ R) := by
  sorry"""

    print("CODE:", v1)
    result1 = prover.run_command(v1)
    print(f"STATE: {result1.proof_state}")

    # Version 2: Split the biconditional
    print("\n📄 VERSION 2: Split biconditional")
    v2 = """theorem distrib_v2 (P Q R : Prop) : (P ∧ Q) ∨ (P ∧ R) ↔ P ∧ (Q ∨ R) := by
  constructor
  · -- Forward direction: (P ∧ Q) ∨ (P ∧ R) → P ∧ (Q ∨ R)
    sorry
  · -- Backward direction: P ∧ (Q ∨ R) → (P ∧ Q) ∨ (P ∧ R)
    sorry"""

    print("CODE:", v2)
    result2 = prover.run_command(v2)
    print(f"STATE: {result2.proof_state}")

    # Version 3: Expand forward direction
    print("\n📄 VERSION 3: Expand forward direction")
    v3 = """theorem distrib_v3 (P Q R : Prop) : (P ∧ Q) ∨ (P ∧ R) ↔ P ∧ (Q ∨ R) := by
  constructor
  · -- Forward direction: (P ∧ Q) ∨ (P ∧ R) → P ∧ (Q ∨ R)
    intro h
    constructor
    · -- Show P
      cases h with
      | inl hpq => exact hpq.left
      | inr hpr => exact hpr.left
    · -- Show Q ∨ R
      cases h with
      | inl hpq => left; exact hpq.right
      | inr hpr => right; exact hpr.right
  · -- Backward direction: P ∧ (Q ∨ R) → (P ∧ Q) ∨ (P ∧ R)
    sorry"""

    print("CODE:", v3)
    result3 = prover.run_command(v3)
    print(f"STATE: {result3.proof_state}")

    # Version 4: Complete proof
    print("\n📄 VERSION 4: Complete proof")
    v4 = """theorem distrib_v4 (P Q R : Prop) : (P ∧ Q) ∨ (P ∧ R) ↔ P ∧ (Q ∨ R) := by
  constructor
  · -- Forward direction: (P ∧ Q) ∨ (P ∧ R) → P ∧ (Q ∨ R)
    intro h
    constructor
    · -- Show P
      cases h with
      | inl hpq => exact hpq.left
      | inr hpr => exact hpr.left
    · -- Show Q ∨ R
      cases h with
      | inl hpq => left; exact hpq.right
      | inr hpr => right; exact hpr.right
  · -- Backward direction: P ∧ (Q ∨ R) → (P ∧ Q) ∨ (P ∧ R)
    intro h
    have hp : P := h.left
    have hqr : Q ∨ R := h.right
    cases hqr with
    | inl hq => left; exact ⟨hp, hq⟩
    | inr hr => right; exact ⟨hp, hr⟩"""

    print("CODE:", v4)
    result4 = prover.run_command(v4)
    print(f"STATE: {result4.proof_state}")
    print(f"SUCCESS: {result4.success}")

    if result4.success:
        print("   🎉 Complete proof successful!")

def demonstrate_practical_workflow():
    """Show how an LLM would practically use this for editing."""
    print("\n" + "=" * 70)
    print("PRACTICAL LLM EDITING WORKFLOW")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n🤖 LLM THOUGHT PROCESS:")
    print("1. Start with goal")
    print("2. Create basic structure")
    print("3. Add more details iteratively")
    print("4. Each version gives fresh proof states to work with")
    print("5. Can mix Command (create versions) + ProofStep (fill details)")
    print()

    # The pattern: Command → ProofStep → Command → ProofStep → ...
    print("📋 THE WINNING PATTERN:")
    print("   Command (create version) → ProofStep (work on details)")
    print("   → Command (create new version) → ProofStep (work on details)")
    print("   → ... until proof is complete")
    print()

    # Example of the pattern
    print("🔄 DEMONSTRATING THE PATTERN:")
    print()

    # Step 1: Command - create version with structure
    print("STEP 1: Command - Create structured version")
    version = """theorem pattern_demo (a b : ℕ) : a + b = b + a := by
  have comm : ∀ x y : ℕ, x + y = y + x := by sorry
  exact comm a b"""

    result = prover.run_command(version)
    print(f"   Created proof state: {result.proof_state}")

    # Step 2: ProofStep - work on the sorry
    if result.proof_state is not None:
        print("STEP 2: ProofStep - Work on the sorry")
        step = prover.run_proof_step(result.proof_state, "intro x y; exact Nat.add_comm x y")
        print(f"   ProofStep success: {step.success}")
        print(f"   Proof complete: {step.proof_complete}")

        if step.proof_complete:
            print("   🎉 PATTERN SUCCESSFUL!")

    print("\n✅ KEY TAKEAWAYS:")
    print("• Don't try to 'edit' existing theorems")
    print("• CREATE new versions with Command")
    print("• WORK on proof states with ProofStep")
    print("• Each Command gives fresh proof states")
    print("• Perfect for iterative LLM development!")

if __name__ == "__main__":
    demonstrate_editing_pattern()
    demonstrate_complex_editing()
    demonstrate_practical_workflow()

    print("\n" + "=" * 70)
    print("🎯 SKETCH EDITING MASTERY:")
    print("=" * 70)
    print()
    print("✅ THE EDITING SECRET:")
    print("   lean-interact doesn't have 'edit' operations")
    print("   → Instead: CREATE new versions of theorems")
    print("   → Each version gets fresh proof states")
    print("   → Work on these states with ProofStep")
    print()
    print("✅ THE WINNING WORKFLOW:")
    print("   1. Command: Create theorem version")
    print("   2. ProofStep: Fill in details")
    print("   3. Command: Create improved version")
    print("   4. ProofStep: Fill more details")
    print("   5. Repeat until proof complete")
    print()
    print("✅ PERFECT FOR LLMs:")
    print("   • LLM analyzes goal")
    print("   • Creates initial structure (Command)")
    print("   • Works on details (ProofStep)")
    print("   • Refines structure (new Command)")
    print("   • Continues until complete")
    print()
    print("🎉 You now have complete sketch editing mastery!")
    print("=" * 70)
