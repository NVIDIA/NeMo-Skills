#!/usr/bin/env python3
"""
Clear Step-by-Step Demo

This shows the core concepts clearly with working examples:
(a) Source code and Lean messages after each step
(b) Iterative proof sketch building from just the theorem
"""

from nemo_skills.code_execution.lean_prover import LeanProver

def demo_clear_source_evolution():
    """Show source code and Lean messages evolution clearly."""
    print("=" * 70)
    print("(A) SOURCE CODE & LEAN MESSAGES - STEP BY STEP")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n📝 INITIAL SKETCH WITH MULTIPLE SORRIES:")
    print("-" * 50)

    sketch = """theorem clear_demo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have get_P : P ∧ Q → P := by sorry
  have get_Q : P ∧ Q → Q := by sorry
  have make_pair : P → Q → Q ∧ P := by sorry
  intro h
  exact make_pair (get_P h) (get_Q h)"""

    print("LEAN SOURCE CODE:")
    print(sketch)
    print()

    result = prover.run_command(sketch)
    print("LEAN COMPILER RESPONSE:")
    print(f"✓ Sketch created: {result.proof_state is not None}")
    print(f"✓ Proof state ID: {result.proof_state}")
    print(f"✓ Has sorry: {result.has_sorry}")
    print(f"✓ Lean says: {result.response}")

    if result.proof_state is not None:
        current_state = result.proof_state

        print("\n" + "=" * 70)
        print("STEP 1: Fill in 'get_P : P ∧ Q → P'")
        print("=" * 70)

        step1 = prover.run_proof_step(current_state, "intro pq; exact pq.1")
        print(f"TACTIC: intro pq; exact pq.1")
        print(f"SUCCESS: {step1.success}")
        print(f"NEW PROOF STATE: {step1.proof_state}")
        print(f"LEAN RESPONSE: {step1.response}")

        if step1.success:
            print("\n" + "=" * 70)
            print("STEP 2: Fill in 'get_Q : P ∧ Q → Q'")
            print("=" * 70)

            step2 = prover.run_proof_step(step1.proof_state, "intro pq; exact pq.2")
            print(f"TACTIC: intro pq; exact pq.2")
            print(f"SUCCESS: {step2.success}")
            print(f"NEW PROOF STATE: {step2.proof_state}")
            print(f"LEAN RESPONSE: {step2.response}")

            if step2.success:
                print("\n" + "=" * 70)
                print("STEP 3: Fill in 'make_pair : P → Q → Q ∧ P'")
                print("=" * 70)

                step3 = prover.run_proof_step(step2.proof_state, "intro p q; exact ⟨q, p⟩")
                print(f"TACTIC: intro p q; exact ⟨q, p⟩")
                print(f"SUCCESS: {step3.success}")
                print(f"PROOF COMPLETE: {step3.proof_complete}")
                print(f"LEAN RESPONSE: {step3.response}")

                if step3.proof_complete:
                    print("\n🎉 PROOF COMPLETED! All sorries filled in successfully!")

def demo_iterative_building():
    """Show iterative building from just the theorem."""
    print("\n" + "=" * 70)
    print("(B) ITERATIVE BUILDING FROM THEOREM")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n🏗️ PHASE 1: Start with just the theorem")
    print("-" * 50)

    phase1_code = "theorem iterative (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"
    print("LEAN CODE:")
    print(phase1_code)

    result1 = prover.run_command(phase1_code)
    print(f"\nRESULT: Created proof state {result1.proof_state}")
    print(f"LEAN: {result1.response}")

    print("\n🏗️ PHASE 2: Add first decomposition")
    print("-" * 50)

    phase2_code = """theorem iterative2 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have extract_P : P ∧ Q → P := by sorry
  intro h
  sorry -- will use extract_P"""

    print("LEAN CODE:")
    print(phase2_code)

    result2 = prover.run_command(phase2_code)
    print(f"\nRESULT: Created proof state {result2.proof_state}")
    print(f"LEAN: {result2.response}")

    print("\n🏗️ PHASE 3: Add more structure")
    print("-" * 50)

    phase3_code = """theorem iterative3 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have extract_P : P ∧ Q → P := by sorry
  have extract_Q : P ∧ Q → Q := by sorry
  intro h
  sorry -- will combine extract_P and extract_Q"""

    print("LEAN CODE:")
    print(phase3_code)

    result3 = prover.run_command(phase3_code)
    print(f"\nRESULT: Created proof state {result3.proof_state}")
    print(f"LEAN: {result3.response}")

    print("\n🏗️ PHASE 4: Complete the structure")
    print("-" * 50)

    phase4_code = """theorem iterative4 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have extract_P : P ∧ Q → P := by sorry
  have extract_Q : P ∧ Q → Q := by sorry
  intro h
  exact ⟨extract_Q h, extract_P h⟩"""

    print("LEAN CODE:")
    print(phase4_code)

    result4 = prover.run_command(phase4_code)
    print(f"\nRESULT: Created proof state {result4.proof_state}")
    print(f"LEAN: {result4.response}")

    print("\n🏗️ PHASE 5: Fill in the sorries")
    print("-" * 50)

    if result4.proof_state is not None:
        print("Filling extract_P...")
        step1 = prover.run_proof_step(result4.proof_state, "intro pq; exact pq.1")
        print(f"  Success: {step1.success}")

        if step1.success:
            print("Filling extract_Q...")
            step2 = prover.run_proof_step(step1.proof_state, "intro pq; exact pq.2")
            print(f"  Success: {step2.success}")
            print(f"  Proof complete: {step2.proof_complete}")

            if step2.proof_complete:
                print("  🎉 ITERATIVE BUILD COMPLETE!")

def demo_proof_state_tracking():
    """Show how proof states are tracked and evolve."""
    print("\n" + "=" * 70)
    print("(C) PROOF STATE EVOLUTION TRACKING")
    print("=" * 70)

    prover = LeanProver(mathlib_enabled=True)

    print("\n🔍 TRACKING PROOF STATES:")

    # Show how proof states evolve with each addition
    stages = [
        ("Basic theorem", "theorem track (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"),
        ("Add 1 have", """theorem track2 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  sorry"""),
        ("Add 2 haves", """theorem track3 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  sorry"""),
        ("Complete structure", """theorem track4 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩""")
    ]

    for stage_name, code in stages:
        print(f"\n📊 {stage_name.upper()}:")
        result = prover.run_command(code)
        print(f"   Proof state ID: {result.proof_state}")
        print(f"   Has sorry: {result.has_sorry}")
        print(f"   Number of sorry references: {code.count('sorry')}")
        print(f"   Lean response: {result.response}")

if __name__ == "__main__":
    demo_clear_source_evolution()
    demo_iterative_building()
    demo_proof_state_tracking()

    print("\n" + "=" * 70)
    print("📋 KEY INSIGHTS:")
    print("=" * 70)
    print()
    print("✅ (A) SOURCE CODE EVOLUTION:")
    print("   • You can see exact Lean source code at each step")
    print("   • Lean compiler gives clear feedback (success/warnings)")
    print("   • Proof state IDs track progress (0 → 1 → 2 → ...)")
    print("   • Each filled sorry advances the proof state")
    print()
    print("✅ (B) ITERATIVE SKETCH BUILDING:")
    print("   • Start: theorem name : statement := by sorry")
    print("   • Phase 2: Add have statements gradually")
    print("   • Phase 3: Build up more structure")
    print("   • Phase 4: Complete the proof framework")
    print("   • Phase 5: Fill in all sorry statements")
    print()
    print("✅ (C) PROOF STATE TRACKING:")
    print("   • Each have statement creates new proof obligations")
    print("   • Proof state IDs increment as structure grows")
    print("   • Can work on any sorry in any order using ProofStep")
    print("   • Perfect for LLM-driven incremental development")
    print()
    print("🎯 This workflow enables:")
    print("   • LLM analyzes goal → creates skeleton → fills details")
    print("   • Incremental proof development and refinement")
    print("   • Easy backtracking and modification")
    print("   • Structured approach to complex proofs")
    print("=" * 70)
