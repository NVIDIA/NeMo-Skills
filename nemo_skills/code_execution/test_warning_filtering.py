#!/usr/bin/env python3

"""
Test script to verify that harmless warnings are filtered correctly.
"""

from nemo_skills.code_execution.lean_interact_session import LeanInteractSession

def test_warning_filtering():
    """Test that elan warnings are filtered out properly"""

    # Create a session instance to test the filtering method
    session = LeanInteractSession("theorem test : True := by sorry")

    # Test various stderr scenarios
    test_cases = [
        {
            "input": "warning: failed to query latest release, using existing version 'leanprover/lean4:v4.20.0'\nActual error message",
            "expected": "Actual error message",
            "description": "Elan warning with actual error"
        },
        {
            "input": "warning: failed to query latest release, using existing version 'leanprover/lean4:v4.20.0'\n",
            "expected": "",
            "description": "Only elan warning"
        },
        {
            "input": "Some real error message\nAnother error line",
            "expected": "Some real error message\nAnother error line",
            "description": "Real errors should pass through"
        },
        {
            "input": "",
            "expected": "",
            "description": "Empty stderr"
        },
        {
            "input": "warning: failed to query latest release, using existing version 'leanprover/lean4:v4.15.0'\nusing existing version 'leanprover/lean4:v4.15.0'\nReal error after warnings",
            "expected": "Real error after warnings",
            "description": "Multiple warning lines with real error"
        }
    ]

    print("Testing warning filtering...")

    for i, test_case in enumerate(test_cases):
        result = session._filter_harmless_warnings(test_case["input"])

        print(f"\n--- Test Case {i+1}: {test_case['description']} ---")
        print(f"Input: {repr(test_case['input'])}")
        print(f"Expected: {repr(test_case['expected'])}")
        print(f"Got: {repr(result)}")

        if result == test_case["expected"]:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            return False

    print("\n✅ All warning filtering tests passed!")
    session.cleanup()
    return True

def test_real_scenario():
    """Test a realistic scenario with the elan warning"""
    print("\n=== Testing Real Scenario ===")

    # This demonstrates how the warning would be handled in practice
    session = LeanInteractSession("theorem test_example : True := by sorry")

    # Simulate what would happen with elan warnings
    mock_stderr = "warning: failed to query latest release, using existing version 'leanprover/lean4:v4.20.0'\n"

    filtered = session._filter_harmless_warnings(mock_stderr)
    print(f"Original stderr: {repr(mock_stderr)}")
    print(f"Filtered stderr: {repr(filtered)}")

    # This should be empty (no error) so tactics would be marked as successful
    is_clean = not filtered.strip()
    print(f"Would be treated as success: {is_clean}")

    session.cleanup()
    return is_clean

def test_guaranteed_proof():
    """Test an actual proof that should always work"""
    print("\n=== Testing Guaranteed Proof ===")

    try:
        # Create a session with a guaranteed-to-work theorem
        session = LeanInteractSession("theorem test_guaranteed : True := by sorry")

        # Try to apply the guaranteed-to-work tactic
        result = session.apply_tactic("trivial")

        print(f"Applied 'trivial' to 'True':")
        print(f"  Success: {result['success']}")
        print(f"  Complete: {result['proof_complete']}")
        print(f"  Error: {result.get('error', 'None')}")

        if result['success'] and result['proof_complete']:
            print("✅ Guaranteed proof test PASSED")
            script = session.get_proof_script()
            print(f"  Generated script: {script}")
        else:
            print("❌ Guaranteed proof test FAILED")
            print("  This indicates a problem with the Lean setup or LeanInteractSession")

        session.cleanup()
        return result['success'] and result['proof_complete']

    except Exception as e:
        print(f"❌ Error during guaranteed proof test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Lean Warning Filtering")
    print("=" * 50)

    try:
                # Run the tests
        filtering_passed = test_warning_filtering()
        scenario_passed = test_real_scenario()
        proof_passed = test_guaranteed_proof()

        if filtering_passed and scenario_passed and proof_passed:
            print("\n🎉 All tests passed! Elan warnings will be handled correctly and proofs work as expected.")
        else:
            print("\n❌ Some tests failed.")
            if not filtering_passed:
                print("  - Warning filtering failed")
            if not scenario_passed:
                print("  - Real scenario test failed")
            if not proof_passed:
                print("  - Guaranteed proof test failed")

    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
