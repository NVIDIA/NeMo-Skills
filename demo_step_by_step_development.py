#!/usr/bin/env python3
"""
Step-by-Step Proof Development Demo

This shows:
(a) The actual Lean source code and compiler messages after each proof step
(b) How to iteratively build a proof sketch starting from just the theorem
"""

from nemo_skills.code_execution.lean_prover import LeanProver

def demo_source_code_evolution():
    """Show what the source code and Lean messages look like after each step."""
    print("=" * 80)
    print("(A) SOURCE CODE & LEAN MESSAGES EVOLUTION")
    print("=" * 80)

    prover = LeanProver(mathlib_enabled=True)

    # Start with a complete sketch
    print("\n🎯 INITIAL SKETCH:")
    print("-" * 50)

    initial_sketch = """theorem step_by_step_demo (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  have h5 : P → Q → R → P ∧ (Q ∧ R) := by sorry
  intro h
  exact h5 (h3 (h1 h)) (h4 (h1 h)) (h2 h)"""

    print("LEAN CODE:")
    print(initial_sketch)

    result = prover.run_command(initial_sketch)
    print(f"\nLEAN COMPILER RESPONSE:")
    print(f"Success: {result.success}")
    print(f"Has sorry: {result.has_sorry}")
    print(f"Proof state ID: {result.proof_state}")
    print(f"Response: {result.response}")
    print(f"Number of sorries: {result.response.count('sorry') if result.response else 0}")

    if result.proof_state is not None:
        current_state = result.proof_state

        # Step 1: Work on h1
        print("\n" + "=" * 80)
        print("STEP 1: Working on h1: (P ∧ Q) ∧ R → P ∧ Q")
        print("=" * 80)

        step1 = prover.run_proof_step(current_state, "intro h; exact h.left")
        print(f"TACTIC APPLIED: intro h; exact h.left")
        print(f"SUCCESS: {step1.success}")
        print(f"NEW PROOF STATE: {step1.proof_state}")
        print(f"LEAN RESPONSE: {step1.response}")
        if step1.goals:
            print(f"REMAINING GOALS: {len(step1.goals)}")
            for i, goal in enumerate(step1.goals):
                print(f"  Goal {i+1}: {goal[:100]}..." if len(goal) > 100 else f"  Goal {i+1}: {goal}")

        if step1.success:
            current_state = step1.proof_state

            # Step 2: Work on h2
            print("\n" + "=" * 80)
            print("STEP 2: Working on h2: (P ∧ Q) ∧ R → R")
            print("=" * 80)

            step2 = prover.run_proof_step(current_state, "intro h; exact h.right")
            print(f"TACTIC APPLIED: intro h; exact h.right")
            print(f"SUCCESS: {step2.success}")
            print(f"NEW PROOF STATE: {step2.proof_state}")
            print(f"LEAN RESPONSE: {step2.response}")
            if step2.goals:
                print(f"REMAINING GOALS: {len(step2.goals)}")
                for i, goal in enumerate(step2.goals):
                    print(f"  Goal {i+1}: {goal[:100]}..." if len(goal) > 100 else f"  Goal {i+1}: {goal}")

            if step2.success:
                current_state = step2.proof_state

                # Step 3: Work on h3
                print("\n" + "=" * 80)
                print("STEP 3: Working on h3: P ∧ Q → P")
                print("=" * 80)

                step3 = prover.run_proof_step(current_state, "intro h; exact h.left")
                print(f"TACTIC APPLIED: intro h; exact h.left")
                print(f"SUCCESS: {step3.success}")
                print(f"NEW PROOF STATE: {step3.proof_state}")
                print(f"LEAN RESPONSE: {step3.response}")
                if step3.goals:
                    print(f"REMAINING GOALS: {len(step3.goals)}")

                if step3.success:
                    current_state = step3.proof_state

                    # Step 4: Work on h4
                    print("\n" + "=" * 80)
                    print("STEP 4: Working on h4: P ∧ Q → Q")
                    print("=" * 80)

                    step4 = prover.run_proof_step(current_state, "intro h; exact h.right")
                    print(f"TACTIC APPLIED: intro h; exact h.right")
                    print(f"SUCCESS: {step4.success}")
                    print(f"NEW PROOF STATE: {step4.proof_state}")
                    print(f"LEAN RESPONSE: {step4.response}")
                    if step4.goals:
                        print(f"REMAINING GOALS: {len(step4.goals)}")

                    if step4.success:
                        # Step 5: Work on h5
                        print("\n" + "=" * 80)
                        print("STEP 5: Working on h5: P → Q → R → P ∧ (Q ∧ R)")
                        print("=" * 80)

                        step5 = prover.run_proof_step(step4.proof_state, "intro hp hq hr; exact ⟨hp, ⟨hq, hr⟩⟩")
                        print(f"TACTIC APPLIED: intro hp hq hr; exact ⟨hp, ⟨hq, hr⟩⟩")
                        print(f"SUCCESS: {step5.success}")
                        print(f"PROOF COMPLETE: {step5.proof_complete}")
                        print(f"LEAN RESPONSE: {step5.response}")

                        if step5.proof_complete:
                            print("\n🎉 FULL PROOF COMPLETED!")
                        else:
                            print(f"REMAINING GOALS: {len(step5.goals) if step5.goals else 0}")

def demo_iterative_sketch_building():
    """Show how to build up a proof sketch iteratively starting from just the theorem."""
    print("\n" + "=" * 80)
    print("(B) ITERATIVE SKETCH BUILDING FROM THEOREM")
    print("=" * 80)

    prover = LeanProver(mathlib_enabled=True)

    # Phase 1: Start with just the theorem
    print("\n🏗️ PHASE 1: Start with basic theorem")
    print("-" * 50)

    phase1 = "theorem iterative_build (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by sorry"

    print("LEAN CODE:")
    print(phase1)

    result1 = prover.run_command(phase1)
    print(f"\nLEAN RESPONSE:")
    print(f"Success: {result1.success}")
    print(f"Proof state: {result1.proof_state}")
    print(f"Response: {result1.response}")

    # Phase 2: Add first decomposition
    print("\n🏗️ PHASE 2: Add first level of decomposition")
    print("-" * 50)

    phase2 = """theorem iterative_build2 (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have extract_left : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have extract_right : (P ∧ Q) ∧ R → R := by sorry
  intro h
  sorry -- will combine extract_left and extract_right"""

    print("LEAN CODE:")
    print(phase2)

    result2 = prover.run_command(phase2)
    print(f"\nLEAN RESPONSE:")
    print(f"Success: {result2.success}")
    print(f"Proof state: {result2.proof_state}")
    print(f"Response: {result2.response}")

    # Phase 3: Further decompose
    print("\n🏗️ PHASE 3: Further decompose the subgoals")
    print("-" * 50)

    phase3 = """theorem iterative_build3 (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have extract_left : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have extract_right : (P ∧ Q) ∧ R → R := by sorry
  have get_P : P ∧ Q → P := by sorry
  have get_Q : P ∧ Q → Q := by sorry
  intro h
  sorry -- will use get_P, get_Q, extract_left, extract_right"""

    print("LEAN CODE:")
    print(phase3)

    result3 = prover.run_command(phase3)
    print(f"\nLEAN RESPONSE:")
    print(f"Success: {result3.success}")
    print(f"Proof state: {result3.proof_state}")
    print(f"Response: {result3.response}")

    # Phase 4: Complete the structure
    print("\n🏗️ PHASE 4: Complete the proof structure")
    print("-" * 50)

    phase4 = """theorem iterative_build4 (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have extract_left : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have extract_right : (P ∧ Q) ∧ R → R := by sorry
  have get_P : P ∧ Q → P := by sorry
  have get_Q : P ∧ Q → Q := by sorry
  have combine : P → Q → R → P ∧ (Q ∧ R) := by sorry
  intro h
  exact combine (get_P (extract_left h)) (get_Q (extract_left h)) (extract_right h)"""

    print("LEAN CODE:")
    print(phase4)

    result4 = prover.run_command(phase4)
    print(f"\nLEAN RESPONSE:")
    print(f"Success: {result4.success}")
    print(f"Proof state: {result4.proof_state}")
    print(f"Response: {result4.response}")

    # Phase 5: Now fill in the sorries
    print("\n🏗️ PHASE 5: Fill in the sorry statements")
    print("-" * 50)

    if result4.proof_state is not None:
        current_state = result4.proof_state

        print("Working on extract_left...")
        step1 = prover.run_proof_step(current_state, "intro h; exact h.left")
        print(f"Applied: intro h; exact h.left -> Success: {step1.success}")

        if step1.success:
            print("Working on extract_right...")
            step2 = prover.run_proof_step(step1.proof_state, "intro h; exact h.right")
            print(f"Applied: intro h; exact h.right -> Success: {step2.success}")

            if step2.success:
                print("Working on get_P...")
                step3 = prover.run_proof_step(step2.proof_state, "intro h; exact h.left")
                print(f"Applied: intro h; exact h.left -> Success: {step3.success}")

                if step3.success:
                    print("Working on get_Q...")
                    step4 = prover.run_proof_step(step3.proof_state, "intro h; exact h.right")
                    print(f"Applied: intro h; exact h.right -> Success: {step4.success}")

                    if step4.success:
                        print("Working on combine...")
                        step5 = prover.run_proof_step(step4.proof_state, "intro hp hq hr; exact ⟨hp, ⟨hq, hr⟩⟩")
                        print(f"Applied: intro hp hq hr; exact ⟨hp, ⟨hq, hr⟩⟩ -> Success: {step5.success}")
                        print(f"PROOF COMPLETE: {step5.proof_complete}")

                        if step5.proof_complete:
                            print("\n🎉 ITERATIVELY BUILT PROOF IS COMPLETE!")

def demo_llm_thought_process():
    """Show how an LLM might think through this process."""
    print("\n" + "=" * 80)
    print("(C) LLM THOUGHT PROCESS SIMULATION")
    print("=" * 80)

    print("\n🤖 LLM ANALYSIS:")
    print("Goal: Prove (P ∧ Q) ∧ R → P ∧ (Q ∧ R)")
    print("This is about reassociating conjunctions")
    print()

    print("🤖 LLM STRATEGY:")
    print("1. I need to break apart the left side: (P ∧ Q) ∧ R")
    print("2. I need to build the right side: P ∧ (Q ∧ R)")
    print("3. Key insight: I need P, Q, R separately to recombine them")
    print()

    print("🤖 LLM CREATES SKELETON:")

    prover = LeanProver(mathlib_enabled=True)

    llm_skeleton = """theorem llm_proof (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  -- LLM identifies key steps:
  have step1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry -- extract left part
  have step2 : (P ∧ Q) ∧ R → R := by sorry       -- extract right part
  have step3 : P ∧ Q → P := by sorry             -- get P from left
  have step4 : P ∧ Q → Q := by sorry             -- get Q from left
  -- Now I have P, Q, R and need to combine as P ∧ (Q ∧ R)
  intro h
  exact ⟨step3 (step1 h), ⟨step4 (step1 h), step2 h⟩⟩"""

    print(llm_skeleton)

    result = prover.run_command(llm_skeleton)
    print(f"\n🤖 LLM SKELETON RESULT:")
    print(f"Created successfully: {result.proof_state is not None}")
    print(f"Response: {result.response}")

    print("\n🤖 LLM FILLS IN DETAILS:")
    print("Now I'll work on each sorry systematically...")

if __name__ == "__main__":
    demo_source_code_evolution()
    demo_iterative_sketch_building()
    demo_llm_thought_process()

    print("\n" + "=" * 80)
    print("📋 SUMMARY OF WORKFLOWS:")
    print("=" * 80)
    print()
    print("✅ (A) SOURCE CODE EVOLUTION:")
    print("   • Start with complete sketch (multiple have...sorry)")
    print("   • Each proof step shows exact Lean compiler response")
    print("   • Track proof state IDs as they evolve")
    print("   • See remaining goals after each step")
    print()
    print("✅ (B) ITERATIVE SKETCH BUILDING:")
    print("   • Phase 1: theorem name : statement := by sorry")
    print("   • Phase 2: Add first level have statements")
    print("   • Phase 3: Decompose further as needed")
    print("   • Phase 4: Complete structure")
    print("   • Phase 5: Fill in all sorry statements")
    print()
    print("✅ (C) LLM INTEGRATION:")
    print("   • Analyze goal mathematically")
    print("   • Identify key proof steps")
    print("   • Create structured skeleton")
    print("   • Fill in details systematically")
    print()
    print("🎯 Both workflows perfectly support LLM-driven theorem proving!")
    print("=" * 80)
