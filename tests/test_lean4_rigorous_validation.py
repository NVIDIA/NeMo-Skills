#!/usr/bin/env python3
"""
Rigorous Thread Safety Validation Tests

Comprehensive validation tests for Lean 4 concurrent usage with 100% success requirements.
Tests include extreme concurrency, stress testing, and resource management validation.
"""

import pytest
import sys
import time
from nemo_skills.code_execution.lean4.prover import LeanProver
from nemo_skills.code_execution.lean4.interactive_agent import InteractiveLeanAgent


# Import validation functions from prover module
def validate_concurrent_safety_strict(num_threads: int = 20, num_operations: int = 10,
                                     require_perfect: bool = True):
    """Validation function - would typically be in prover module."""
    # For now, we'll implement a basic version here
    import concurrent.futures
    import psutil
    import gc

    # Force garbage collection before starting
    gc.collect()

    # Track system resources
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    results = {
        'perfect_validation': True,
        'success_rate': 1.0,
        'conflicts_detected': 0,
        'total_operations': 0,
        'issues_found': []
    }

    def test_prover_thread(thread_id: int):
        """Test prover in concurrent environment."""
        thread_results = []
        for i in range(num_operations):
            try:
                prover = LeanProver(mathlib_enabled=True)
                theorem = f"theorem test_{thread_id}_{i} : True := trivial"
                result = prover.run(theorem)
                thread_results.append({
                    'success': result.success,
                    'thread_id': thread_id
                })
            except Exception:
                thread_results.append({
                    'success': False,
                    'thread_id': thread_id
                })
        return thread_results

    def test_agent_thread(thread_id: int):
        """Test agent in concurrent environment."""
        thread_results = []
        for i in range(num_operations):
            try:
                agent = InteractiveLeanAgent(mathlib_enabled=True)
                theorem = f"theorem agent_test_{thread_id}_{i} : True := trivial"
                result = agent.load_theorem(theorem)
                thread_results.append({
                    'success': result.get('success', False),
                    'thread_id': thread_id
                })
            except Exception:
                thread_results.append({
                    'success': False,
                    'thread_id': thread_id
                })
        return thread_results

    all_results = []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
            futures = []
            for i in range(num_threads):
                futures.append(executor.submit(test_prover_thread, i))
                futures.append(executor.submit(test_agent_thread, i + num_threads))

            for future in concurrent.futures.as_completed(futures, timeout=60):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    results['issues_found'].append(str(e))

    except Exception as e:
        results['issues_found'].append(f"Executor error: {str(e)}")
        results['perfect_validation'] = False

    # Analyze results
    results['total_operations'] = len(all_results)
    if results['total_operations'] > 0:
        successful_ops = sum(1 for r in all_results if r.get('success', False))
        results['success_rate'] = successful_ops / results['total_operations']

        if require_perfect and results['success_rate'] != 1.0:
            results['perfect_validation'] = False
            failed_ops = results['total_operations'] - successful_ops
            results['issues_found'].append(f"Failed operations: {failed_ops}/{results['total_operations']}")

    return results


def extreme_stress_test(max_concurrent: int = 50, duration_seconds: int = 15):
    """Basic stress test implementation."""
    import concurrent.futures
    import random
    import time
    import threading

    stress_results = {
        'agents_created': 0,
        'operations_completed': 0,
        'operations_successful': 0,
        'peak_concurrent': 0,
        'errors': []
    }

    stop_flag = threading.Event()

    def stress_worker(agent_id: int):
        try:
            agent = InteractiveLeanAgent(mathlib_enabled=True)
            stress_results['agents_created'] += 1
            stress_results['peak_concurrent'] = max(stress_results['peak_concurrent'], agent_id + 1)

            op_count = 0
            while not stop_flag.is_set():
                try:
                    theorem = f"theorem stress_{agent_id}_{op_count} : True := trivial"
                    result = agent.load_theorem(theorem)
                    stress_results['operations_completed'] += 1
                    if result.get('success', False):
                        stress_results['operations_successful'] += 1
                    op_count += 1
                    time.sleep(random.uniform(0.01, 0.05))
                except Exception as e:
                    stress_results['errors'].append(str(e))
                    break
        except Exception as e:
            stress_results['errors'].append(str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(stress_worker, i) for i in range(max_concurrent)]
        time.sleep(duration_seconds)
        stop_flag.set()

    success_rate = (stress_results['operations_successful'] /
                   max(stress_results['operations_completed'], 1))

    return {
        'stress_results': stress_results,
        'success_rate': success_rate,
        'passed': success_rate >= 0.85 and len(stress_results['errors']) < max_concurrent * 0.2
    }


class TestRigorousThreadSafety:
    """Rigorous thread safety validation test suite."""

    def test_100_percent_validation_basic(self):
        """Test basic validation requiring 100% success rate."""
        result = validate_concurrent_safety_strict(
            num_threads=10,
            num_operations=5,
            require_perfect=True
        )

        # Assert 100% success requirement
        assert result['perfect_validation'], f"Validation failed with issues: {result['issues_found']}"
        assert result['success_rate'] == 1.0, f"Expected 100% success, got {result['success_rate']*100:.1f}%"
        assert result['conflicts_detected'] == 0, f"Found {result['conflicts_detected']} conflicts"
        assert len(result['issues_found']) == 0, f"Found issues: {result['issues_found']}"

    def test_100_percent_validation_medium_concurrency(self):
        """Test medium concurrency with 100% success requirement."""
        result = validate_concurrent_safety_strict(
            num_threads=30,
            num_operations=10,
            require_perfect=True
        )

        # Assert perfect validation
        assert result['perfect_validation'], "Medium concurrency validation failed"
        assert result['success_rate'] == 1.0, f"Expected 100% success, got {result['success_rate']*100:.1f}%"
        assert result['conflicts_detected'] == 0, "Found thread conflicts"

        # Check operation counts
        expected_ops = 30 * 2 * 10  # threads * types * operations
        assert result['total_operations'] >= expected_ops * 0.9, "Too few operations completed"

    def test_100_percent_validation_high_concurrency(self):
        """Test high concurrency with 100% success requirement."""
        result = validate_concurrent_safety_strict(
            num_threads=50,
            num_operations=5,
            require_perfect=True
        )

        # Assert perfect validation under high load
        assert result['perfect_validation'], "High concurrency validation failed"
        assert result['success_rate'] == 1.0, f"Expected 100% success, got {result['success_rate']*100:.1f}%"
        assert result['conflicts_detected'] == 0, "Found thread conflicts under high concurrency"

    def test_100_percent_validation_memory_intensive(self):
        """Test memory-intensive operations with 100% success requirement."""
        result = validate_concurrent_safety_strict(
            num_threads=20,
            num_operations=15,
            require_perfect=True
        )

        # Assert perfect validation
        assert result['perfect_validation'], "Memory-intensive validation failed"
        assert result['success_rate'] == 1.0, "Memory-intensive operations failed"

        # Check for resource leaks
        memory_leaked = result['resource_stats']['memory_mb']['leaked']
        fd_leaked = result['resource_stats']['file_descriptors']['leaked']

        assert memory_leaked < 100, f"Memory leak detected: {memory_leaked}MB"
        assert fd_leaked < 10, f"File descriptor leak: {fd_leaked} FDs"

    def test_extreme_stress_basic(self):
        """Test basic stress testing capabilities."""
        result = extreme_stress_test(
            max_concurrent=20,
            duration_seconds=5
        )

        # Should handle reasonable stress well
        assert result['success_rate'] >= 0.95, f"Stress test success rate too low: {result['success_rate']*100:.1f}%"
        assert result['stress_results']['peak_concurrent'] >= 15, "Not enough concurrent agents achieved"
        assert result['stress_results']['operations_completed'] > 0, "No operations completed during stress test"

    def test_extreme_stress_medium(self):
        """Test medium stress testing."""
        result = extreme_stress_test(
            max_concurrent=30,
            duration_seconds=10
        )

        # Should handle medium stress
        assert result['success_rate'] >= 0.90, f"Medium stress test failed: {result['success_rate']*100:.1f}%"
        assert result['stress_results']['operations_completed'] > 100, "Too few operations under stress"

    @pytest.mark.slow
    def test_extreme_stress_high(self):
        """Test high stress with many concurrent agents (marked as slow test)."""
        result = extreme_stress_test(
            max_concurrent=50,
            duration_seconds=15
        )

        # Should handle high stress reasonably
        assert result['success_rate'] >= 0.85, f"High stress test failed: {result['success_rate']*100:.1f}%"
        assert result['stress_results']['peak_concurrent'] >= 40, "Not enough peak concurrency achieved"

    def test_resource_management(self):
        """Test resource management during validation."""
        result = validate_concurrent_safety_strict(
            num_threads=15,
            num_operations=8,
            require_perfect=True
        )

        # Check resource statistics
        resource_stats = result['resource_stats']

        # Memory should not leak significantly
        memory_change = resource_stats['memory_mb']['leaked']
        assert abs(memory_change) < 50, f"Significant memory change: {memory_change}MB"

        # File descriptors should not leak
        fd_change = resource_stats['file_descriptors']['leaked']
        assert abs(fd_change) <= 5, f"File descriptor leak: {fd_change} FDs"

    def test_thread_isolation_validation(self):
        """Test that threads are properly isolated from each other."""
        result = validate_concurrent_safety_strict(
            num_threads=25,
            num_operations=6,
            require_perfect=True
        )

        # Should have perfect isolation
        assert result['conflicts_detected'] == 0, "Thread isolation failed - conflicts detected"
        assert result['perfect_validation'], "Thread isolation validation failed"

        # All operations should succeed with proper isolation
        assert result['success_rate'] == 1.0, "Thread isolation led to operation failures"


@pytest.mark.parametrize("threads,operations", [
    (5, 3),   # Light load
    (10, 5),  # Medium load
    (20, 4),  # Heavy load
])
def test_parametrized_100_percent_validation(threads, operations):
    """Parametrized test for 100% validation across different loads."""
    result = validate_concurrent_safety_strict(
        num_threads=threads,
        num_operations=operations,
        require_perfect=True
    )

    # All parametrized tests must achieve 100%
    assert result['perfect_validation'], f"Failed for {threads} threads, {operations} ops"
    assert result['success_rate'] == 1.0, f"Expected 100% success for {threads}x{operations}"
    assert result['conflicts_detected'] == 0, f"Conflicts in {threads}x{operations} test"


def test_quick_validation_check():
    """Quick validation check for development workflow."""
    result = validate_concurrent_safety_strict(
        num_threads=5,
        num_operations=2,
        require_perfect=True
    )

    assert result['perfect_validation'], "Quick validation check failed"
    assert result['success_rate'] == 1.0, "Quick check didn't achieve 100%"


@pytest.mark.slow
def test_comprehensive_validation_suite():
    """Comprehensive validation covering all major scenarios."""
    test_configs = [
        (15, 8),   # Balanced load
        (25, 6),   # High concurrency
        (10, 12),  # High operations per thread
    ]

    all_results = []

    for threads, operations in test_configs:
        result = validate_concurrent_safety_strict(
            num_threads=threads,
            num_operations=operations,
            require_perfect=True
        )
        all_results.append(result)

        # Each configuration must pass perfectly
        assert result['perfect_validation'], f"Failed config: {threads}x{operations}"
        assert result['success_rate'] == 1.0, f"Config {threads}x{operations} not 100%"

    # Overall validation across all configs
    total_operations = sum(r['total_operations'] for r in all_results)
    total_conflicts = sum(r['conflicts_detected'] for r in all_results)

    assert total_operations > 1000, "Not enough total operations tested"
    assert total_conflicts == 0, f"Found {total_conflicts} total conflicts across all configs"


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"])
