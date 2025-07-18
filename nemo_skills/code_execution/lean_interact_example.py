#!/usr/bin/env python3

"""
Example usage of LeanInteractSession for interactive proof development.

This script demonstrates how to use the LeanInteract-based tool for
creating and updating proof states in an interactive manner.

Note: This now uses the official LeanInteract library with TempRequireProject,
providing reliable access to Lean 4 with mathlib and aesop support by default.
"""

import logging
from nemo_skills.code_execution.lean_interact_session import create_lean_interact_session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_simple_proof():
    """Example of a simple proof using LeanInteractSession"""
    print("=== Simple Proof Example ===")

    # Create a session for a simple theorem
    theorem = "theorem example_trivial : True := by sorry"
    session = create_lean_interact_session(theorem)

    print(f"Session ID: {session.session_id}")
    print(f"Theorem: {session.theorem_statement}")

    # Get initial proof state
    state = session.get_proof_state()
    print(f"Initial goals: {state['current_goals']}")

    # Apply a tactic - 'trivial' always works for 'True'
    result = session.apply_tactic("trivial")
    print(f"Applied 'trivial': success={result['success']}, complete={result['proof_complete']}")

    if result['success']:
        print("✅ Proof completed!")
        script = session.get_proof_script()
        print(f"Final proof script:\n{script}")
    else:
        print("❌ Proof failed")
        if result['error']:
            print(f"Error: {result['error']}")

    # Clean up
    session.cleanup()
    return session

def example_mathlib_functionality():
    """Example showcasing mathlib imports and functionality"""
    print("\n=== Mathlib Functionality Example ===")

    # Example using Real numbers from mathlib
    theorem = """theorem example_real_analysis (x : ℝ) (h : x ≥ 0) : Real.sqrt (x^2) = x := by
  sorry"""

    session = create_lean_interact_session(theorem)  # mathlib is now enabled by default via TempRequireProject
    print(f"Session ID: {session.session_id}")
    print("TempRequireProject provides:")
    print("- Mathlib with Real numbers, topology, analysis")
    print("- Aesop automation tactic")
    print("- BigOperators, Natural numbers")
    print("- Reliable dependency management")

    # Get initial state
    state = session.get_proof_state()
    print(f"Initial goals: {state['current_goals']}")

    # Apply mathlib tactics
    result1 = session.apply_tactic("rw [Real.sqrt_sq h]")
    print(f"Applied Real.sqrt_sq: success={result1['success']}")

    if result1['success']:
        print(f"✅ Proof completed with mathlib!")
        script = session.get_proof_script()
        print(f"Final proof script:\n{script}")
    else:
        print("❌ Proof failed with mathlib tactics")
        if result1['error']:
            print(f"Error: {result1['error']}")

    # Test another mathlib example with aesop
    print("\n--- Aesop automation example ---")
    theorem2 = """theorem example_aesop (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  sorry"""

    session2 = create_lean_interact_session(theorem2)
    result2 = session2.apply_tactic("aesop")
    print(f"Applied aesop: success={result2['success']}")
    if result2['success']:
        print(f"✅ Aesop proof: {session2.get_proof_script()}")
    session2.cleanup()

    # Clean up
    session.cleanup()
    return session

def example_complex_proof():
    """Example of a more complex proof with branching"""
    print("\n=== Complex Proof Example ===")

    # Create a session for a more complex theorem
    theorem = """theorem example_algebra (x : ℝ) (h : x^2 - 3*x + 2 = 0) : x = 1 ∨ x = 2 := by
  sorry"""

    session = create_lean_interact_session(theorem)
    print(f"Session ID: {session.session_id}")

    # Get initial state
    state = session.get_proof_state()
    print(f"Initial goals: {state['current_goals']}")

    # Try first approach
    print("\n--- Trying factoring approach ---")
    result1 = session.apply_tactic("have h1 : (x - 1) * (x - 2) = 0 := by ring_nf; rw [← h]; ring")
    print(f"Applied factoring: success={result1['success']}")

    if result1['success']:
        result2 = session.apply_tactic("have h2 : x - 1 = 0 ∨ x - 2 = 0 := by exact mul_eq_zero.mp h1")
        print(f"Applied mul_eq_zero: success={result2['success']}")

        if result2['success']:
            result3 = session.apply_tactic("cases h2 with | inl h => left; linarith | inr h => right; linarith")
            print(f"Applied cases: success={result3['success']}, complete={result3['proof_complete']}")

    # Create a branch to try alternative approach
    print("\n--- Creating alternative branch ---")
    try:
        alt_branch = session.create_branch("quadratic_formula")
        session.switch_branch(alt_branch)

        # Try alternative approach
        result4 = session.apply_tactic("field_simp at h")
        print(f"Alternative approach: success={result4['success']}")
    except Exception as e:
        print(f"Branching error: {e}")

    # Show analysis
    analysis = session.analyze_proof_attempts()
    print(f"\n--- Analysis ---")
    print(f"Total branches: {analysis['total_branches']}")
    print(f"Total steps: {analysis['total_steps']}")
    print(f"Success rate: {analysis['success_rate']:.2%}")
    print(f"Completed branches: {analysis['completed_branches']}")

    # Show all branches
    print(f"\n--- All branches ---")
    for branch_info in session.list_branches():
        print(f"Branch {branch_info['name']}: {branch_info['step_count']} steps, status={branch_info['status']}")

    # Clean up
    session.cleanup()
    return session

def example_session_persistence():
    """Example of saving and loading sessions"""
    print("\n=== Session Persistence Example ===")

    # Create and work on a proof
    theorem = "theorem example_comm (a b : ℕ) : a + b = b + a := by sorry"
    session = create_lean_interact_session(theorem)

    # Apply some tactics
    result = session.apply_tactic("rw [Nat.add_comm]")
    print(f"Applied Nat.add_comm: success={result['success']}, complete={result['proof_complete']}")

    # Save session
    if session.save_session("example_session.json"):
        print("Session saved to example_session.json")

        # Load session
        try:
            loaded_session = session.load_session("example_session.json")
            print(f"Loaded session ID: {loaded_session.session_id}")

            # Continue work on loaded session
            state = loaded_session.get_proof_state()
            print(f"Loaded session goals: {state['current_goals']}")

            # Clean up
            loaded_session.cleanup()
        except Exception as e:
            print(f"Error loading session: {e}")

    # Clean up
    session.cleanup()
    return session

def example_guaranteed_working_proofs():
    """Examples of proofs that should work with the LeanInteract library"""
    print("\n=== Guaranteed Working Proofs ===")

    # Test 1: True is always provable with trivial
    print("\n--- Test 1: Proving True ---")
    theorem1 = "theorem test_true : True := by sorry"
    session1 = create_lean_interact_session(theorem1)
    result1 = session1.apply_tactic("trivial")
    print(f"'trivial' for True: success={result1['success']}, complete={result1['proof_complete']}")
    if result1['success']:
        print(f"✅ Proof: {session1.get_proof_script()}")
    session1.cleanup()

    # Test 2: Reflexivity for natural numbers
    print("\n--- Test 2: Reflexivity ---")
    theorem2 = "theorem test_rfl (n : ℕ) : n = n := by sorry"
    session2 = create_lean_interact_session(theorem2)
    result2 = session2.apply_tactic("rfl")
    print(f"'rfl' for n = n: success={result2['success']}, complete={result2['proof_complete']}")
    if result2['success']:
        print(f"✅ Proof: {session2.get_proof_script()}")
    session2.cleanup()

    # Test 3: Simple arithmetic with mathlib
    print("\n--- Test 3: Simple arithmetic ---")
    theorem3 = "theorem test_arith : 1 + 1 = 2 := by sorry"
    session3 = create_lean_interact_session(theorem3)
    result3 = session3.apply_tactic("norm_num")
    print(f"'norm_num' for 1 + 1 = 2: success={result3['success']}, complete={result3['proof_complete']}")
    if result3['success']:
        print(f"✅ Proof: {session3.get_proof_script()}")
    session3.cleanup()

    # Test 4: Constructor
    print("\n--- Test 4: Constructor ---")
    theorem4 = "theorem test_constructor (A B : Prop) (ha : A) (hb : B) : A ∧ B := by sorry"
    session4 = create_lean_interact_session(theorem4)
    result4 = session4.apply_tactic("constructor; exact ha; exact hb")
    print(f"'constructor; exact ha; exact hb' for A ∧ B: success={result4['success']}, complete={result4['proof_complete']}")
    if result4['success']:
        print(f"✅ Proof: {session4.get_proof_script()}")
    session4.cleanup()

    # Test 5: Real numbers with mathlib
    print("\n--- Test 5: Real numbers ---")
    theorem5 = "theorem test_real (x : ℝ) : x = x := by sorry"
    session5 = create_lean_interact_session(theorem5)
    result5 = session5.apply_tactic("rfl")
    print(f"'rfl' for Real numbers: success={result5['success']}, complete={result5['proof_complete']}")
    if result5['success']:
        print(f"✅ Proof: {session5.get_proof_script()}")
    session5.cleanup()

    # Count successes
    successes = sum([result1['success'], result2['success'], result3['success'], result4['success'], result5['success']])
    print(f"\n🎯 Summary: {successes}/5 guaranteed proofs succeeded")

    return successes == 5

def example_integration_with_sandbox():
    """Example of how this could integrate with existing sandbox patterns"""
    print("\n=== Integration Example ===")

    # This shows how the LeanInteractSession could be used alongside
    # existing sandbox execution patterns

    theorem = "theorem example_simple : True := by sorry"
    session = create_lean_interact_session(theorem)

    # The session provides a higher-level interface than direct sandbox execution
    # It maintains state, supports branching, and provides comprehensive tracking

    print(f"Session created with ID: {session.session_id}")
    print(f"Using LeanInteract library: {hasattr(session, 'server')}")
    print(f"Mathlib support: {session.use_mathlib}")
    print(f"Auto-recovery server: {session.use_auto_server}")

    # Apply a simple tactic
    result = session.apply_tactic("trivial")
    print(f"Simple tactic result: {result['success']}")

    # Clean up
    session.cleanup()
    return session

def test_error_handling():
    """Test error handling and recovery"""
    print("\n=== Error Handling Test ===")

    theorem = "theorem test_error (n : ℕ) : n = n + 1 := by sorry"
    session = create_lean_interact_session(theorem)

    # This should fail
    result = session.apply_tactic("rfl")
    print(f"Attempting impossible proof: success={result['success']}")
    if not result['success']:
        print(f"Expected error: {result['error']}")

    # Now try a different approach that should also fail
    result2 = session.apply_tactic("simp")
    print(f"Trying simp on false goal: success={result2['success']}")

    # Check session state after errors
    state = session.get_proof_state()
    print(f"Session status after errors: {state['status']}")

    session.cleanup()
    return session

if __name__ == "__main__":
    try:
        # Run all examples
        print("🚀 LeanInteractSession Examples (Using Official LeanInteract Library)")
        print("=" * 70)

        example_simple_proof()
        guaranteed_success = example_guaranteed_working_proofs()
        example_mathlib_functionality()
        example_complex_proof()
        example_session_persistence()
        example_integration_with_sandbox()
        test_error_handling()

        if guaranteed_success:
            print("\n✅ All examples completed successfully!")
            print("🎯 All guaranteed proofs worked - your LeanInteract setup is working correctly!")
        else:
            print("\n⚠️  Some guaranteed proofs failed - there may be issues with your setup.")
            print("Check that the LeanInteract library is properly installed.")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
