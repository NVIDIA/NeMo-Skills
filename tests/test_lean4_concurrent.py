#!/usr/bin/env python3
"""
Comprehensive Concurrent Usage Tests for Lean 4 Prover/Agent

Tests for concurrent access, thread safety, resource management, and performance scaling.
Designed to test up to thousands of concurrent instances safely.

TESTING MODES:
- MOCK_SERVERS=1 (default): Use mock servers for safe, fast testing
- MOCK_SERVERS=0: Use real Lean servers for actual performance testing
"""

import asyncio
import concurrent.futures
import gc
import os
import psutil
import pytest
import threading
import time
from contextlib import contextmanager
from typing import List, Dict, Any, Callable, Optional
from unittest.mock import Mock, patch

from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent

# Configuration: Set to False to use real Lean servers
USE_MOCK_SERVERS = os.getenv('MOCK_SERVERS', '1') != '0'
MAX_REAL_CONCURRENT = int(os.getenv('MAX_REAL_CONCURRENT', '10'))  # Limit for real servers

if USE_MOCK_SERVERS:
    print("🔧 Using MOCK SERVERS for safe concurrent testing")
else:
    print("⚡ Using REAL LEAN SERVERS - actual performance testing")
    print(f"   Real server concurrency limited to {MAX_REAL_CONCURRENT}")


class ResourceMonitor:
    """Monitor system resources during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.initial_open_files = len(self.process.open_files())
        self.max_memory = self.initial_memory
        self.max_open_files = self.initial_open_files
        self.monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start background resource monitoring."""
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory = self.process.memory_info().rss
                open_files = len(self.process.open_files())
                self.max_memory = max(self.max_memory, memory)
                self.max_open_files = max(self.max_open_files, open_files)
                time.sleep(0.1)  # Monitor every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        try:
            current_memory = self.process.memory_info().rss
            current_open_files = len(self.process.open_files())
            return {
                "memory_mb": {
                    "initial": self.initial_memory / 1024 / 1024,
                    "current": current_memory / 1024 / 1024,
                    "max": self.max_memory / 1024 / 1024,
                    "increase": (current_memory - self.initial_memory) / 1024 / 1024
                },
                "open_files": {
                    "initial": self.initial_open_files,
                    "current": current_open_files,
                    "max": self.max_open_files,
                    "increase": current_open_files - self.initial_open_files
                }
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"error": "Could not access process information"}


@contextmanager
def lean_server_context(mock_enabled=None, **mock_kwargs):
    """Context manager that either mocks AutoLeanServer or uses real servers."""
    if mock_enabled is None:
        mock_enabled = USE_MOCK_SERVERS

    if mock_enabled:
        # Use mock server for safe testing
        class ConfigurableMockServer:
            """Mock server with configurable behavior for testing."""

            def __init__(self, config, latency_ms=1.0, failure_rate=0.0):
                self.config = config
                self.latency_ms = mock_kwargs.get('latency_ms', latency_ms)
                self.failure_rate = mock_kwargs.get('failure_rate', failure_rate)
                self.call_count = 0
                self.thread_id = threading.get_ident()

            def run(self, command):
                """Mock run method that simulates Lean execution."""
                self.call_count += 1
                current_thread = threading.get_ident()

                # Simulate processing time
                time.sleep(self.latency_ms / 1000.0)

                # Simulate failures if configured
                import random
                if random.random() < self.failure_rate:
                    raise Exception(f"Simulated failure (rate: {self.failure_rate})")

                # Mock response based on command type
                if hasattr(command, 'cmd'):
                    # Command execution
                    cmd = command.cmd
                    if 'theorem' in cmd and 'sorry' in cmd:
                        return Mock(
                            env=self.call_count,
                            sorries=[Mock(proof_state=self.call_count * 10)],
                            messages=[],
                            success=True
                        )
                    else:
                        return Mock(
                            env=self.call_count,
                            sorries=[],
                            messages=[],
                            success=True
                        )
                else:
                    # ProofStep execution
                    return Mock(
                        proof_status="Complete" if self.call_count % 3 == 0 else "Incomplete",
                        proof_state=self.call_count * 10 + 1,
                        goals=["test goal"],
                        success=True
                    )

        with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', ConfigurableMockServer):
            yield "mock"
    else:
        # Use real Lean servers - no mocking
        print(f"🔥 WARNING: Using REAL Lean servers - this will spawn actual Lean processes!")
        yield "real"


class MockAutoLeanServer:
    """Legacy mock server - use lean_server_context() instead."""

    def __init__(self, config):
        self.config = config
        self.call_count = 0
        self.thread_id = threading.get_ident()

    def run(self, command):
        """Mock run method that simulates Lean execution."""
        self.call_count += 1

        # Simulate some work
        time.sleep(0.001)  # 1ms delay to simulate real work

        # Mock response based on command type
        if hasattr(command, 'cmd'):
            cmd = command.cmd
            if 'theorem' in cmd and 'sorry' in cmd:
                return Mock(
                    env=self.call_count,
                    sorries=[Mock(proof_state=self.call_count * 10)],
                    messages=[],
                    success=True
                )
            else:
                return Mock(
                    env=self.call_count,
                    sorries=[],
                    messages=[],
                    success=True
                )
        else:
            return Mock(
                proof_status="Complete" if self.call_count % 3 == 0 else "Incomplete",
                proof_state=self.call_count * 10 + 1,
                goals=["test goal"],
                success=True
            )


@pytest.fixture
def mock_lean_server():
    """Fixture to mock AutoLeanServer to avoid spawning real processes."""
    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', MockAutoLeanServer):
        yield


@contextmanager
def resource_monitor():
    """Context manager for resource monitoring."""
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def get_safe_concurrency_level(requested_level):
    """Get safe concurrency level based on testing mode."""
    if USE_MOCK_SERVERS:
        return requested_level
    else:
        # Limit concurrency for real servers to avoid overwhelming system
        return min(requested_level, MAX_REAL_CONCURRENT)


# Basic Concurrent Functionality Tests

@pytest.mark.parametrize("num_concurrent", [5, 10, 20])
def test_concurrent_prover_creation(num_concurrent):
    """Test creating multiple LeanProver instances concurrently."""
    safe_concurrent = get_safe_concurrency_level(num_concurrent)

    with resource_monitor() as monitor:
        with lean_server_context(latency_ms=1.0):

            def create_prover():
                return LeanProver(mathlib_enabled=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=safe_concurrent) as executor:
                futures = [executor.submit(create_prover) for _ in range(safe_concurrent)]
                provers = [future.result() for future in concurrent.futures.as_completed(futures)]

            assert len(provers) == safe_concurrent
            assert all(isinstance(p, LeanProver) for p in provers)

            # Check that each prover has its own server instance
            server_ids = [id(p.server) for p in provers]
            assert len(set(server_ids)) == safe_concurrent  # All unique

            stats = monitor.get_stats()
            print(f"Created {safe_concurrent} provers - Memory increase: {stats['memory_mb']['increase']:.2f}MB")

            if not USE_MOCK_SERVERS:
                print(f"⚡ REAL SERVER TEST: {safe_concurrent} actual Lean processes created")


@pytest.mark.parametrize("num_concurrent", [5, 10, 20])
def test_concurrent_interactive_agent_creation(num_concurrent):
    """Test creating multiple InteractiveLeanAgent instances concurrently."""
    safe_concurrent = get_safe_concurrency_level(num_concurrent)

    with resource_monitor() as monitor:
        with lean_server_context(latency_ms=2.0):

            def create_agent():
                return InteractiveLeanAgent(mathlib_enabled=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=safe_concurrent) as executor:
                futures = [executor.submit(create_agent) for _ in range(safe_concurrent)]
                agents = [future.result() for future in concurrent.futures.as_completed(futures)]

            assert len(agents) == safe_concurrent
            assert all(isinstance(a, InteractiveLeanAgent) for a in agents)

            stats = monitor.get_stats()
            print(f"Created {safe_concurrent} agents - Memory increase: {stats['memory_mb']['increase']:.2f}MB")

            if not USE_MOCK_SERVERS:
                print(f"⚡ REAL SERVER TEST: {safe_concurrent} actual Lean interactive agents created")


def test_concurrent_theorem_execution():
    """Test executing theorems concurrently on multiple provers."""
    num_provers = get_safe_concurrency_level(10)
    theorems_per_prover = 3 if USE_MOCK_SERVERS else 1  # Fewer theorems for real servers

    with resource_monitor() as monitor:
        with lean_server_context(latency_ms=5.0):  # Slightly higher latency for realism

            provers = [LeanProver(mathlib_enabled=True) for _ in range(num_provers)]

            def execute_theorems(prover, prover_id):
                results = []
                for i in range(theorems_per_prover):
                    theorem = f"theorem test_{prover_id}_{i} : True := trivial"
                    result = prover.run(theorem)
                    results.append((prover_id, i, result.success))
                return results

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_provers) as executor:
                futures = [
                    executor.submit(execute_theorems, prover, i)
                    for i, prover in enumerate(provers)
                ]
                all_results = []
                for future in concurrent.futures.as_completed(futures):
                    all_results.extend(future.result())

            assert len(all_results) == num_provers * theorems_per_prover
            assert all(success for _, _, success in all_results)

            stats = monitor.get_stats()
            print(f"Executed {len(all_results)} theorems concurrently - "
                  f"Memory increase: {stats['memory_mb']['increase']:.2f}MB")

            if not USE_MOCK_SERVERS:
                print(f"⚡ REAL SERVER TEST: {len(all_results)} actual theorem proofs completed")


# Real Server Specific Tests

@pytest.mark.skipif(USE_MOCK_SERVERS, reason="Real server test - set MOCK_SERVERS=0 to run")
def test_real_server_performance_baseline():
    """Establish performance baseline with real Lean servers."""
    print("🔥 REAL SERVER PERFORMANCE BASELINE TEST")

    # Test single threaded performance first
    with resource_monitor() as monitor:
        prover = LeanProver(mathlib_enabled=True)

        start_time = time.time()
        theorems = [
            "theorem baseline_1 : True := trivial",
            "theorem baseline_2 : 1 + 1 = 2 := rfl",
            "theorem baseline_3 : ∀ x : Nat, x + 0 = x := fun x => rfl",
        ]

        results = []
        for theorem in theorems:
            result = prover.run(theorem)
            results.append(result.success)

        single_thread_time = time.time() - start_time

    # Test concurrent performance
    concurrent_level = min(3, MAX_REAL_CONCURRENT)
    with resource_monitor() as monitor:
        start_time = time.time()

        def run_theorem(theorem):
            prover = LeanProver(mathlib_enabled=True)
            result = prover.run(theorem)
            return result.success

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(run_theorem, theorem) for theorem in theorems]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        concurrent_time = time.time() - start_time

        stats = monitor.get_stats()

        print(f"📊 REAL SERVER PERFORMANCE:")
        print(f"   Single-threaded: {single_thread_time:.2f}s for {len(theorems)} theorems")
        print(f"   Concurrent ({concurrent_level}): {concurrent_time:.2f}s for {len(theorems)} theorems")
        print(f"   Speedup: {single_thread_time/concurrent_time:.2f}x")
        print(f"   Memory usage: {stats['memory_mb']['increase']:.1f}MB")

        assert all(results), "Single-threaded results should all succeed"
        assert all(concurrent_results), "Concurrent results should all succeed"
        # Real Lean processes may not always be faster due to startup overhead
        assert concurrent_time < single_thread_time * 3.0, "Concurrent should not be more than 3x slower"


@pytest.mark.skipif(USE_MOCK_SERVERS, reason="Real server test - set MOCK_SERVERS=0 to run")
def test_real_server_resource_usage():
    """Test actual resource usage with real Lean servers."""
    print("🔥 REAL SERVER RESOURCE USAGE TEST")

    concurrent_level = min(5, MAX_REAL_CONCURRENT)

    with resource_monitor() as monitor:
        def heavy_theorem_work():
            try:
                prover = LeanProver(mathlib_enabled=True)
                # Use simpler theorem that's guaranteed to work
                result = prover.run("theorem resource_test : ∀ (n : Nat), n + 0 = n := fun n => rfl")
                return result.success
            except Exception as e:
                print(f"Real server error: {e}")
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(heavy_theorem_work) for _ in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        stats = monitor.get_stats()

        print(f"📊 REAL SERVER RESOURCE USAGE:")
        print(f"   Concurrent level: {concurrent_level}")
        print(f"   Success rate: {sum(results)}/{len(results)}")
        print(f"   Peak memory: {stats['memory_mb']['max']:.1f}MB")
        print(f"   Memory increase: {stats['memory_mb']['increase']:.1f}MB")
        print(f"   Open files: {stats['open_files']['max']}")

        # For real servers, we expect most to succeed but some might fail due to system limits
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.6, f"Success rate too low: {success_rate*100:.1f}% - check system resources"

        # Real servers should show actual memory usage
        if success_rate > 0:  # Only check if some operations succeeded
            assert stats['memory_mb']['increase'] >= 0, "Should show actual memory usage"


# Mock vs Real Comparison Test

def test_mock_vs_real_comparison():
    """Compare mock vs real server behavior (when both are available)."""
    if USE_MOCK_SERVERS:
        pytest.skip("This test compares modes - run with different MOCK_SERVERS settings")

    print("🔄 MOCK VS REAL COMPARISON TEST")

    # This test would run both modes and compare results
    # For now, just verify real servers work
    prover = LeanProver(mathlib_enabled=True)
    result = prover.run("theorem comparison_test : True := trivial")

    print(f"✅ Real server verification: success={result.success}")
    assert result.success, "Real server should work for basic theorem"


def test_concurrent_interactive_editing(mock_lean_server):
    """Test concurrent interactive editing on multiple agents."""
    num_agents = 8

    with resource_monitor() as monitor:
        agents = [InteractiveLeanAgent(mathlib_enabled=True) for _ in range(num_agents)]

        def interactive_session(agent, agent_id):
            # Load theorem
            theorem = f"theorem interactive_{agent_id} (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"
            result = agent.load_theorem(theorem)

            # Edit clauses
            edit_results = []
            if result.get("editable_clauses"):
                for clause_id in result["editable_clauses"]:
                    if "sorry" in clause_id:
                        edit_result = agent.edit_clause(clause_id, "intro h; exact ⟨h.right, h.left⟩")
                        edit_results.append(edit_result.get("edit_successful", False))

            return agent_id, len(edit_results), all(edit_results) if edit_results else True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [
                executor.submit(interactive_session, agent, i)
                for i, agent in enumerate(agents)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == num_agents
        assert all(success for _, _, success in results)

        stats = monitor.get_stats()
        print(f"Completed {num_agents} interactive sessions - "
              f"Memory increase: {stats['memory_mb']['increase']:.2f}MB")


# Thread Safety Tests

def test_thread_safety_single_prover():
    """Test thread safety when multiple threads use the same prover."""
    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', MockAutoLeanServer):
        prover = LeanProver(mathlib_enabled=True)
        num_threads = 10
        operations_per_thread = 5

        def worker(thread_id):
            results = []
            for i in range(operations_per_thread):
                theorem = f"theorem thread_{thread_id}_{i} : True := trivial"
                result = prover.run(theorem)
                results.append(result.success)
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        assert len(all_results) == num_threads * operations_per_thread
        # Note: Without proper thread safety, this might fail in real usage
        # The mock ensures success, but real usage might need synchronization


def test_thread_safety_shared_state():
    """Test concurrent access to shared state in provers."""
    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', MockAutoLeanServer):
        prover = LeanProver(mathlib_enabled=True)

        def start_and_work_on_proof(proof_id):
            proof_name = f"proof_{proof_id}"
            result = prover.start_proof(proof_name, ": True")

            if result.success:
                # Apply some tactics
                tactic_result = prover.apply_tactic_to_proof(proof_name, "trivial")
                return proof_name, tactic_result.success
            return proof_name, False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(start_and_work_on_proof, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        proof_names = [name for name, _ in results]
        successes = [success for _, success in results]

        assert len(set(proof_names)) == 5  # All unique proof names
        assert all(successes)  # All should succeed with mock


# Performance Scaling Tests

@pytest.mark.parametrize("scale", [10, 25, 50, 100])
def test_performance_scaling_creation(mock_lean_server, scale):
    """Test performance scaling for prover creation."""
    times = []
    memory_usage = []

    for num_provers in [scale // 4, scale // 2, scale]:
        with resource_monitor() as monitor:
            start_time = time.time()

            def create_prover():
                return LeanProver(mathlib_enabled=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_provers, 20)) as executor:
                futures = [executor.submit(create_prover) for _ in range(num_provers)]
                provers = [future.result() for future in concurrent.futures.as_completed(futures)]

            end_time = time.time()
            times.append((num_provers, end_time - start_time))

            stats = monitor.get_stats()
            memory_usage.append((num_provers, stats['memory_mb']['increase']))

            # Force cleanup
            del provers
            gc.collect()

    # Print scaling results
    print("\nPerformance Scaling Results:")
    print("Provers | Time (s) | Memory (MB)")
    for (num, t), (_, mem) in zip(times, memory_usage):
        print(f"{num:7} | {t:8.2f} | {mem:10.2f}")

    # Basic scaling assertions
    assert all(t < 60 for _, t in times), "Creation should not take more than 60 seconds"
    assert all(mem < 1000 for _, mem in memory_usage), "Memory increase should be reasonable"


@pytest.mark.parametrize("num_provers,operations_per_prover", [(20, 5), (50, 3), (100, 2)])
def test_performance_scaling_execution(mock_lean_server, num_provers, operations_per_prover):
    """Test performance scaling for theorem execution."""
    with resource_monitor() as monitor:
        start_time = time.time()

        # Create provers
        provers = [LeanProver(mathlib_enabled=True) for _ in range(num_provers)]
        creation_time = time.time()

        # Execute operations
        def execute_operations(prover, prover_id):
            results = []
            for i in range(operations_per_prover):
                theorem = f"theorem perf_{prover_id}_{i} : True := trivial"
                result = prover.run(theorem)
                results.append(result.success)
            return sum(results)

        max_workers = min(num_provers, 50)  # Limit concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(execute_operations, prover, i)
                for i, prover in enumerate(provers)
            ]
            successful_operations = sum(
                future.result() for future in concurrent.futures.as_completed(futures)
            )

        end_time = time.time()
        total_operations = num_provers * operations_per_prover

        stats = monitor.get_stats()

        print(f"\nScaling Test: {num_provers} provers × {operations_per_prover} ops")
        print(f"Total operations: {total_operations}")
        print(f"Successful operations: {successful_operations}")
        print(f"Creation time: {creation_time - start_time:.2f}s")
        print(f"Execution time: {end_time - creation_time:.2f}s")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Operations per second: {successful_operations / (end_time - creation_time):.2f}")
        print(f"Memory usage: {stats['memory_mb']['increase']:.2f}MB")

        assert successful_operations == total_operations
        assert end_time - start_time < 120  # Should complete within 2 minutes


# Stress Tests

@pytest.mark.slow
def test_stress_high_concurrency(mock_lean_server):
    """Stress test with high concurrency (run with pytest -m slow)."""
    num_concurrent = 200  # High concurrency

    with resource_monitor() as monitor:
        def quick_operation():
            prover = LeanProver(mathlib_enabled=True)
            result = prover.run("theorem stress_test : True := trivial")
            return result.success

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(quick_operation) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()

        stats = monitor.get_stats()

        print(f"\nStress Test Results:")
        print(f"Operations: {num_concurrent}")
        print(f"Success rate: {sum(results)}/{len(results)} ({sum(results)/len(results)*100:.1f}%)")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Operations per second: {num_concurrent / (end_time - start_time):.2f}")
        print(f"Memory increase: {stats['memory_mb']['increase']:.2f}MB")

        assert sum(results) == num_concurrent  # All should succeed with mock


@pytest.mark.slow
def test_stress_memory_usage(mock_lean_server):
    """Test memory usage under stress (run with pytest -m slow)."""
    iterations = 50
    provers_per_iteration = 10

    memory_history = []

    with resource_monitor() as monitor:
        for i in range(iterations):
            # Create provers
            provers = [LeanProver(mathlib_enabled=True) for _ in range(provers_per_iteration)]

            # Do some work
            for prover in provers:
                prover.run(f"theorem stress_memory_{i} : True := trivial")

            # Record memory usage
            stats = monitor.get_stats()
            memory_history.append(stats['memory_mb']['current'])

            # Clean up
            del provers
            gc.collect()

            # Report progress
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations}, Memory: {stats['memory_mb']['current']:.1f}MB")

    final_stats = monitor.get_stats()

    print(f"\nMemory Stress Test Results:")
    print(f"Iterations: {iterations}")
    print(f"Initial memory: {final_stats['memory_mb']['initial']:.2f}MB")
    print(f"Final memory: {final_stats['memory_mb']['current']:.2f}MB")
    print(f"Max memory: {final_stats['memory_mb']['max']:.2f}MB")
    print(f"Memory increase: {final_stats['memory_mb']['increase']:.2f}MB")

    # Memory should not grow indefinitely
    final_increase = final_stats['memory_mb']['increase']
    assert final_increase < 500, f"Memory increase too high: {final_increase:.2f}MB"


# Error Handling Tests

def test_concurrent_error_handling():
    """Test error handling under concurrent load."""
    num_concurrent = 20

    with resource_monitor() as monitor:
        if USE_MOCK_SERVERS:
            # Use the mock context with high failure rate to ensure we get some failures
            with lean_server_context(latency_ms=1.0, failure_rate=0.25):  # 25% failure rate

                def operation_with_error_handling():
                    try:
                        prover = LeanProver(mathlib_enabled=True)
                        result = prover.run("theorem error_test : True := trivial")
                        return "success" if result.success else "failure"
                    except Exception as e:
                        return f"error: {str(e)[:20]}..."

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(operation_with_error_handling) for _ in range(num_concurrent)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]

                successes = sum(1 for r in results if r == "success")
                failures = sum(1 for r in results if r == "failure")
                errors = sum(1 for r in results if r.startswith("error"))

                print(f"\nMock Server Error Handling Results:")
                print(f"Total operations: {num_concurrent}")
                print(f"Successes: {successes}")
                print(f"Failures: {failures}")
                print(f"Errors: {errors}")
                print(f"Error rate: {(failures + errors)/num_concurrent*100:.1f}%")

                assert successes > 0, "Should have some successes"
                assert (errors + failures) > 0, "Should have some errors or failures (by design)"

        else:
            # Real server mode - test error handling with problematic theorems
            print("⚡ REAL SERVER ERROR HANDLING: Testing actual error recovery")

            # Mix of good and problematic theorems
            test_theorems = [
                "theorem good1 : True := trivial",
                "theorem good2 : 1 + 1 = 2 := rfl",
                "theorem bad1 : sorry := sorry",  # This should fail
                "theorem good3 : True := trivial",
                "theorem bad2 : False",  # This should fail - no proof
                "theorem good4 : True := trivial",
            ]

            def test_theorem(theorem):
                try:
                    prover = LeanProver(mathlib_enabled=True)
                    result = prover.run(theorem)
                    return "success" if result.success else "failure"
                except Exception as e:
                    return f"error: {str(e)[:20]}..."

            # Test each theorem type
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, MAX_REAL_CONCURRENT)) as executor:
                futures = [executor.submit(test_theorem, theorem) for theorem in test_theorems]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            successes = sum(1 for r in results if r == "success")
            failures = sum(1 for r in results if r == "failure")
            errors = sum(1 for r in results if r.startswith("error"))

            print(f"\nReal Server Error Handling Results:")
            print(f"Total operations: {len(test_theorems)}")
            print(f"Successes: {successes}")
            print(f"Failures: {failures}")
            print(f"Errors: {errors}")
            print(f"Error rate: {(failures + errors)/len(test_theorems)*100:.1f}%")

            # For real servers, we should have some successes and some failures
            assert successes >= 2, "Should have at least 2 successful theorems"
            assert (errors + failures) >= 1, "Should have at least 1 error/failure from bad theorems"


# Resource Management Tests

def test_resource_cleanup():
    """Test that resources are properly cleaned up."""
    with resource_monitor() as monitor:
        initial_stats = monitor.get_stats()

        # Create and destroy provers multiple times
        for iteration in range(5):
            provers = [LeanProver(mathlib_enabled=True) for _ in range(10)]

            # Use the provers
            for i, prover in enumerate(provers):
                prover.run(f"theorem cleanup_test_{iteration}_{i} : True := trivial")

            # Explicit cleanup
            del provers
            gc.collect()

        final_stats = monitor.get_stats()

        print(f"\nResource Cleanup Test:")
        print(f"Memory increase: {final_stats['memory_mb']['increase']:.2f}MB")
        print(f"Open files increase: {final_stats['open_files']['increase']}")

        # Should not accumulate too many resources
        assert final_stats['memory_mb']['increase'] < 200, "Memory should not accumulate excessively"


@pytest.mark.asyncio
async def test_async_concurrent_usage():
    """Test concurrent usage with asyncio."""
    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', MockAutoLeanServer):

        async def async_prover_operation(prover_id):
            # Simulate async work
            await asyncio.sleep(0.001)  # 1ms delay

            prover = LeanProver(mathlib_enabled=True)
            result = prover.run(f"theorem async_{prover_id} : True := trivial")
            return result.success

        # Run many concurrent async operations
        num_operations = 50
        tasks = [async_prover_operation(i) for i in range(num_operations)]
        results = await asyncio.gather(*tasks)

        assert len(results) == num_operations
        assert all(results)


# Configuration Tests

@pytest.mark.parametrize("mathlib_enabled", [True, False])
def test_concurrent_different_configs(mock_lean_server, mathlib_enabled):
    """Test concurrent usage with different configurations."""
    num_provers = 10

    def create_and_test(prover_id):
        prover = LeanProver(mathlib_enabled=mathlib_enabled)
        result = prover.run(f"theorem config_test_{prover_id} : True := trivial")
        return prover.mathlib_enabled, result.success

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_provers) as executor:
        futures = [executor.submit(create_and_test, i) for i in range(num_provers)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    configs, successes = zip(*results)
    assert all(config == mathlib_enabled for config in configs)
    assert all(successes)


# Utility function for running scaling benchmarks
def run_scaling_benchmark(max_concurrent: int = 1000, step_size: int = 50):
    """
    Run a comprehensive scaling benchmark.

    This function is not a test but a utility for performance analysis.
    Call manually to measure scaling characteristics.
    """
    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', MockAutoLeanServer):
        results = []

        for num_concurrent in range(step_size, max_concurrent + 1, step_size):
            print(f"\nTesting with {num_concurrent} concurrent operations...")

            with resource_monitor() as monitor:
                start_time = time.time()

                def quick_operation():
                    prover = LeanProver(mathlib_enabled=True)
                    return prover.run("theorem benchmark : True := trivial").success

                max_workers = min(num_concurrent, 100)  # Limit thread pool size
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(quick_operation) for _ in range(num_concurrent)]
                    success_count = sum(
                        future.result() for future in concurrent.futures.as_completed(futures)
                    )

                end_time = time.time()
                duration = end_time - start_time
                throughput = success_count / duration

                stats = monitor.get_stats()

                results.append({
                    'concurrent': num_concurrent,
                    'success_count': success_count,
                    'duration': duration,
                    'throughput': throughput,
                    'memory_mb': stats['memory_mb']['increase'],
                    'open_files': stats['open_files']['increase']
                })

                print(f"  Successes: {success_count}/{num_concurrent}")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Throughput: {throughput:.2f} ops/sec")
                print(f"  Memory: +{stats['memory_mb']['increase']:.1f}MB")

        print("\n" + "="*80)
        print("SCALING BENCHMARK RESULTS")
        print("="*80)
        print("Concurrent | Success Rate | Duration(s) | Throughput | Memory(MB)")
        print("-"*80)
        for r in results:
            success_rate = r['success_count'] / r['concurrent'] * 100
            print(f"{r['concurrent']:10} | {success_rate:11.1f}% | {r['duration']:10.2f} | "
                  f"{r['throughput']:9.1f} | {r['memory_mb']:9.1f}")

        return results


if __name__ == "__main__":
    print("Running scaling benchmark...")
    run_scaling_benchmark(max_concurrent=500, step_size=50)
