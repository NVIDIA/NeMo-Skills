#!/usr/bin/env python3
"""
Stress Tests for Lean 4 Prover/Agent - Extreme Concurrent Usage

Tests designed to push the system to its limits with thousands of concurrent instances.
Focuses on resource limits, error recovery, and system stability under extreme load.
"""

import asyncio
import gc
import os
import psutil
import pytest
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent


class SystemResourceTracker:
    """Track system-wide resource usage during stress tests."""

    def __init__(self):
        self.initial_stats = self._get_system_stats()
        self.samples = []
        self.monitoring = False

    def _get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics."""
        try:
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'open_files': len(psutil.Process().open_files()),
                'thread_count': threading.active_count(),
                'process_count': len(psutil.pids())
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'error': 'Could not access system stats'}

    def start_monitoring(self):
        """Start continuous resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                stats = self._get_system_stats()
                stats['timestamp'] = time.time()
                self.samples.append(stats)
                time.sleep(0.5)  # Sample every 500ms
            except Exception:
                break

    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak resource usage during monitoring."""
        if not self.samples:
            return self._get_system_stats()

        peak_memory = max((s.get('memory_percent', 0) for s in self.samples), default=0)
        peak_cpu = max((s.get('cpu_percent', 0) for s in self.samples), default=0)
        max_files = max((s.get('open_files', 0) for s in self.samples), default=0)
        max_threads = max((s.get('thread_count', 0) for s in self.samples), default=0)

        return {
            'peak_memory_percent': peak_memory,
            'peak_cpu_percent': peak_cpu,
            'max_open_files': max_files,
            'max_thread_count': max_threads,
            'initial_stats': self.initial_stats,
            'sample_count': len(self.samples)
        }


# Enhanced mock for stress testing
class StressMockServer:
    """Mock server with configurable stress test behaviors."""

    def __init__(self, config,
                 latency_range: tuple = (0.5, 2.0),
                 failure_rate: float = 0.02,
                 memory_leak_chance: float = 0.0):
        self.config = config
        self.latency_range = latency_range
        self.failure_rate = failure_rate
        self.memory_leak_chance = memory_leak_chance
        self.call_count = 0
        self.thread_id = threading.get_ident()
        self.large_objects = []  # Simulate memory leaks

    def run(self, command):
        """Mock run with realistic stress test behaviors."""
        self.call_count += 1

        # Random latency
        latency = random.uniform(*self.latency_range)
        time.sleep(latency / 1000.0)  # Convert to seconds

        # Random failures
        if random.random() < self.failure_rate:
            raise Exception(f"Stress test failure #{self.call_count}")

        # Simulate occasional memory leaks
        if random.random() < self.memory_leak_chance:
            self.large_objects.append([0] * 10000)  # 10k integers

        # Return mock response
        if hasattr(command, 'cmd'):
            return Mock(
                env=self.call_count,
                sorries=[Mock(proof_state=self.call_count * 10)] if 'sorry' in command.cmd else [],
                messages=[],
                success=True
            )
        else:
            return Mock(
                proof_status="Complete" if self.call_count % 10 != 0 else "Incomplete",
                proof_state=self.call_count * 10,
                goals=["test goal"],
                success=True
            )


@contextmanager
def stress_mock_server(**kwargs):
    """Context manager for stress test mock server."""
    def server_factory(config):
        return StressMockServer(config, **kwargs)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', server_factory):
        yield


# Extreme Concurrency Tests

@pytest.mark.slow
@pytest.mark.parametrize("concurrent_count", [500, 1000, 2000])
def test_extreme_concurrent_prover_creation(concurrent_count):
    """Test creating extreme numbers of provers concurrently."""
    if concurrent_count > 1000 and os.getenv('SKIP_EXTREME_TESTS') == '1':
        pytest.skip("Extreme test skipped (set SKIP_EXTREME_TESTS=0 to run)")

    tracker = SystemResourceTracker()
    tracker.start_monitoring()

    try:
        with stress_mock_server(latency_range=(0.1, 0.5), failure_rate=0.05):

            def create_prover_and_execute():
                try:
                    prover = LeanProver(mathlib_enabled=True)
                    result = prover.run(f"theorem stress_{threading.get_ident()} : True := trivial")
                    return result.success
                except Exception as e:
                    return f"error: {str(e)[:30]}..."

            start_time = time.time()

            # Use reasonable thread pool size to avoid overwhelming system
            max_workers = min(concurrent_count, 200)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(create_prover_and_execute)
                          for _ in range(concurrent_count)]

                results = []
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Progress reporting
                    if completed % (concurrent_count // 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"Completed {completed}/{concurrent_count} "
                              f"({completed/concurrent_count*100:.1f}%) in {elapsed:.1f}s")

            end_time = time.time()

        successes = sum(1 for r in results if r is True)
        errors = len(results) - successes

        print(f"\nExtreme Concurrency Test Results ({concurrent_count} operations):")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Successful operations: {successes}/{concurrent_count} ({successes/concurrent_count*100:.1f}%)")
        print(f"Failed operations: {errors}")
        print(f"Throughput: {successes/(end_time - start_time):.1f} ops/sec")

    finally:
        tracker.stop_monitoring()
        peak_usage = tracker.get_peak_usage()

        print(f"Resource Usage:")
        print(f"  Peak memory: {peak_usage['peak_memory_percent']:.1f}%")
        print(f"  Peak CPU: {peak_usage['peak_cpu_percent']:.1f}%")
        print(f"  Max threads: {peak_usage['max_thread_count']}")
        print(f"  Max files: {peak_usage['max_open_files']}")

    # Should handle extreme load gracefully
    assert successes > concurrent_count * 0.8, f"Success rate too low: {successes/concurrent_count*100:.1f}%"
    assert peak_usage['peak_memory_percent'] < 95, "Memory usage too high"


@pytest.mark.slow
def test_sustained_high_load():
    """Test sustained high load over extended time period."""
    duration_minutes = 2  # 2 minutes of sustained load
    concurrent_level = 100
    operations_per_second = 20

    tracker = SystemResourceTracker()
    tracker.start_monitoring()

    total_operations = 0
    total_successes = 0
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    try:
        with stress_mock_server(latency_range=(0.5, 1.5), failure_rate=0.03):

            def sustained_operation():
                nonlocal total_operations, total_successes
                try:
                    prover = LeanProver(mathlib_enabled=True)
                    result = prover.run("theorem sustained : True := trivial")
                    total_operations += 1
                    if result.success:
                        total_successes += 1
                    return result.success
                except Exception:
                    total_operations += 1
                    return False

            with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
                active_futures = set()

                while time.time() < end_time:
                    # Maintain target operations per second
                    current_time = time.time()
                    elapsed = current_time - start_time
                    target_ops = int(elapsed * operations_per_second)

                    # Submit new operations if needed
                    while len(active_futures) + total_operations < target_ops:
                        if len(active_futures) < concurrent_level:
                            future = executor.submit(sustained_operation)
                            active_futures.add(future)

                    # Check for completed operations
                    completed = set()
                    for future in active_futures:
                        if future.done():
                            completed.add(future)

                    active_futures -= completed

                    # Progress report
                    if int(elapsed) % 10 == 0 and elapsed > 0:
                        current_rate = total_operations / elapsed
                        success_rate = total_successes / total_operations if total_operations > 0 else 0
                        print(f"Minute {elapsed/60:.1f}: {current_rate:.1f} ops/sec, "
                              f"{success_rate*100:.1f}% success")

                    time.sleep(0.1)  # Small delay to prevent tight loop

                # Wait for remaining operations to complete
                for future in active_futures:
                    future.result()

        actual_duration = time.time() - start_time
        actual_rate = total_operations / actual_duration

    finally:
        tracker.stop_monitoring()
        peak_usage = tracker.get_peak_usage()

    print(f"\nSustained Load Test Results:")
    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total operations: {total_operations}")
    print(f"Successful operations: {total_successes}")
    print(f"Success rate: {total_successes/total_operations*100:.1f}%")
    print(f"Average rate: {actual_rate:.1f} ops/sec")
    print(f"Peak memory: {peak_usage['peak_memory_percent']:.1f}%")
    print(f"Peak CPU: {peak_usage['peak_cpu_percent']:.1f}%")

    # Should maintain reasonable performance under sustained load
    assert total_successes > total_operations * 0.85, "Success rate too low under sustained load"
    assert actual_rate > operations_per_second * 0.8, "Throughput too low"
    assert peak_usage['peak_memory_percent'] < 90, "Memory usage unsustainable"


@pytest.mark.slow
def test_resource_exhaustion_recovery():
    """Test system behavior when resources are nearly exhausted."""

    def resource_intensive_operation():
        """Operation that consumes significant resources."""
        # Create multiple provers and agents
        provers = [LeanProver(mathlib_enabled=True) for _ in range(5)]
        agents = [InteractiveLeanAgent(mathlib_enabled=True) for _ in range(3)]

        # Execute operations
        results = []
        for i, prover in enumerate(provers):
            result = prover.run(f"theorem resource_test_{i} : True := trivial")
            results.append(result.success)

        # Interactive operations
        for j, agent in enumerate(agents):
            theorem = f"theorem interactive_{j} : True := by sorry"
            load_result = agent.load_theorem(theorem)
            if load_result.get("editable_clauses"):
                for clause_id in load_result["editable_clauses"]:
                    if "sorry" in clause_id:
                        agent.edit_clause(clause_id, "trivial")

        return sum(results)

    tracker = SystemResourceTracker()
    tracker.start_monitoring()

    try:
        with stress_mock_server(latency_range=(1.0, 3.0), failure_rate=0.1):
            concurrent_levels = [10, 50, 100, 200]  # Gradually increase load

            for level in concurrent_levels:
                print(f"\nTesting resource exhaustion with {level} concurrent operations...")

                start_time = time.time()
                with ThreadPoolExecutor(max_workers=min(level, 100)) as executor:
                    futures = [executor.submit(resource_intensive_operation)
                              for _ in range(level)]

                    successes = 0
                    errors = 0
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            successes += result
                        except Exception as e:
                            errors += 1
                            print(f"Resource exhaustion error: {str(e)[:50]}...")

                duration = time.time() - start_time
                current_stats = tracker._get_system_stats()

                print(f"Level {level}: {successes} successes, {errors} errors, "
                      f"{duration:.1f}s, memory: {current_stats['memory_percent']:.1f}%")

                # Force cleanup between levels
                gc.collect()
                time.sleep(2)

    finally:
        tracker.stop_monitoring()
        peak_usage = tracker.get_peak_usage()

    print(f"\nResource Exhaustion Test Results:")
    print(f"Peak memory usage: {peak_usage['peak_memory_percent']:.1f}%")
    print(f"Peak CPU usage: {peak_usage['peak_cpu_percent']:.1f}%")
    print(f"Max threads: {peak_usage['max_thread_count']}")

    # System should handle resource pressure gracefully
    assert peak_usage['peak_memory_percent'] < 98, "Memory usage too close to system limits"


# Error Recovery and Fault Tolerance Tests

@pytest.mark.slow
def test_high_failure_rate_resilience():
    """Test system resilience under high failure rates."""
    failure_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    concurrent_level = 50
    operations_per_rate = 100

    for failure_rate in failure_rates:
        print(f"\nTesting with {failure_rate*100}% failure rate...")

        with stress_mock_server(latency_range=(0.5, 1.0), failure_rate=failure_rate):

            def resilient_operation():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        prover = LeanProver(mathlib_enabled=True)
                        result = prover.run("theorem resilient : True := trivial")
                        return result.success
                    except Exception as e:
                        if attempt == max_retries - 1:
                            return False
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                return False

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
                futures = [executor.submit(resilient_operation)
                          for _ in range(operations_per_rate)]

                results = [future.result() for future in as_completed(futures)]

            duration = time.time() - start_time
            successes = sum(results)

            expected_success_rate = max(0.1, (1 - failure_rate) ** 0.5)  # With retries
            actual_success_rate = successes / len(results)

            print(f"  Expected success rate: {expected_success_rate*100:.1f}%")
            print(f"  Actual success rate: {actual_success_rate*100:.1f}%")
            print(f"  Duration: {duration:.2f}s")

            # Should achieve reasonable success even with high failure rates
            if failure_rate < 0.8:
                assert actual_success_rate > expected_success_rate * 0.7, \
                    f"Success rate too low for {failure_rate*100}% failure rate"


@pytest.mark.slow
def test_memory_pressure_handling():
    """Test behavior under artificial memory pressure."""

    def memory_pressure_operation():
        """Operation that creates memory pressure."""
        # Create temporary large objects
        large_data = []
        try:
            for i in range(10):
                large_data.append([random.random() for _ in range(10000)])

            prover = LeanProver(mathlib_enabled=True)
            result = prover.run("theorem memory_pressure : True := trivial")
            return result.success

        except MemoryError:
            return False
        finally:
            del large_data  # Cleanup

    tracker = SystemResourceTracker()
    tracker.start_monitoring()

    try:
        # Gradually increase concurrent operations to create memory pressure
        for concurrent_level in [20, 50, 100]:
            print(f"\nTesting memory pressure with {concurrent_level} operations...")

            with stress_mock_server(latency_range=(2.0, 5.0)):  # Longer latency to build pressure

                start_time = time.time()
                with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
                    futures = [executor.submit(memory_pressure_operation)
                              for _ in range(concurrent_level)]

                    results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"Memory pressure exception: {str(e)[:50]}...")
                            results.append(False)

                duration = time.time() - start_time
                successes = sum(results)
                current_memory = psutil.virtual_memory().percent

                print(f"  Successes: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Current memory usage: {current_memory:.1f}%")

                # Force cleanup
                gc.collect()
                time.sleep(1)

    finally:
        tracker.stop_monitoring()
        peak_usage = tracker.get_peak_usage()

    print(f"\nMemory Pressure Test Results:")
    print(f"Peak memory usage: {peak_usage['peak_memory_percent']:.1f}%")

    # Should handle memory pressure without crashing
    assert peak_usage['peak_memory_percent'] < 95, "Memory usage too high"


# Race Condition and Concurrency Safety Tests

def test_concurrent_state_modifications():
    """Test for race conditions in concurrent state modifications."""

    with stress_mock_server(latency_range=(0.1, 0.3)):
        prover = LeanProver(mathlib_enabled=True)

        # Start multiple proofs concurrently
        proof_names = [f"race_proof_{i}" for i in range(20)]

        def start_proof_and_modify(proof_name):
            try:
                # Start proof
                result = prover.start_proof(proof_name, ": True")
                if not result.success:
                    return f"start_failed: {proof_name}"

                # Apply tactics
                for i in range(5):
                    tactic_result = prover.apply_tactic_to_proof(proof_name, f"-- step {i}")
                    if not tactic_result.success:
                        break

                # Try to complete
                completion_result = prover.apply_tactic_to_proof(proof_name, "trivial")
                return completion_result.success

            except Exception as e:
                return f"error: {str(e)[:30]}..."

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(start_proof_and_modify, name)
                      for name in proof_names]

            results = [future.result() for future in as_completed(futures)]

        successes = sum(1 for r in results if r is True)
        errors = sum(1 for r in results if isinstance(r, str))

        print(f"\nConcurrent State Modification Results:")
        print(f"Successful proofs: {successes}/{len(proof_names)}")
        print(f"Errors: {errors}")

        # Some operations should succeed despite potential race conditions
        assert successes > 0, "No operations succeeded - possible deadlock"

        # Check final state consistency
        final_proofs = list(prover.proofs_in_progress.keys())
        print(f"Final proofs in progress: {len(final_proofs)}")


@pytest.mark.asyncio
async def test_async_concurrent_stress():
    """Test extreme async concurrency."""

    async def async_stress_operation(operation_id):
        """Async operation with random delays."""
        await asyncio.sleep(random.uniform(0.01, 0.1))

        # Create prover in async context
        with stress_mock_server(latency_range=(0.1, 0.5)):
            prover = LeanProver(mathlib_enabled=True)
            result = prover.run(f"theorem async_stress_{operation_id} : True := trivial")
            return result.success

    # Test with many async operations
    num_operations = 1000

    start_time = time.time()
    tasks = [async_stress_operation(i) for i in range(num_operations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    successes = sum(1 for r in results if r is True)
    errors = sum(1 for r in results if r is not True)

    print(f"\nAsync Stress Test Results:")
    print(f"Operations: {num_operations}")
    print(f"Successes: {successes}")
    print(f"Errors: {errors}")
    print(f"Duration: {end_time - start_time:.2f}s")
    print(f"Rate: {num_operations / (end_time - start_time):.1f} ops/sec")

    # Should handle high async concurrency
    assert successes > num_operations * 0.8, "Too many async operation failures"


# Utilities for manual stress testing

def run_extended_stress_test(duration_minutes: int = 10, max_concurrent: int = 500):
    """
    Run extended stress test for manual performance analysis.
    Not a pytest test - call manually for long-running stress tests.
    """
    print(f"Running extended stress test: {duration_minutes} minutes, max {max_concurrent} concurrent")

    tracker = SystemResourceTracker()
    tracker.start_monitoring()

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    total_operations = 0
    total_successes = 0

    try:
        with stress_mock_server(latency_range=(1.0, 3.0), failure_rate=0.05):

            def stress_operation():
                nonlocal total_operations, total_successes
                try:
                    prover = LeanProver(mathlib_enabled=True)
                    result = prover.run("theorem extended_stress : True := trivial")
                    total_operations += 1
                    if result.success:
                        total_successes += 1
                    return result.success
                except Exception:
                    total_operations += 1
                    return False

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                active_futures = set()
                last_report = start_time

                while time.time() < end_time:
                    # Maintain load level
                    while len(active_futures) < max_concurrent // 2:
                        future = executor.submit(stress_operation)
                        active_futures.add(future)

                    # Remove completed futures
                    completed = {f for f in active_futures if f.done()}
                    active_futures -= completed

                    # Periodic reporting
                    current_time = time.time()
                    if current_time - last_report > 30:  # Every 30 seconds
                        elapsed_minutes = (current_time - start_time) / 60
                        rate = total_operations / (current_time - start_time)
                        success_rate = total_successes / total_operations if total_operations > 0 else 0

                        current_stats = tracker._get_system_stats()
                        print(f"Minute {elapsed_minutes:.1f}: "
                              f"{rate:.1f} ops/sec, {success_rate*100:.1f}% success, "
                              f"memory: {current_stats['memory_percent']:.1f}%, "
                              f"active: {len(active_futures)}")

                        last_report = current_time

                    time.sleep(0.1)

                # Wait for remaining operations
                for future in active_futures:
                    try:
                        future.result()
                    except Exception:
                        pass

    finally:
        tracker.stop_monitoring()
        peak_usage = tracker.get_peak_usage()

        actual_duration = time.time() - start_time

        print(f"\nExtended Stress Test Results:")
        print(f"Duration: {actual_duration / 60:.1f} minutes")
        print(f"Total operations: {total_operations}")
        print(f"Successful operations: {total_successes}")
        print(f"Success rate: {total_successes/total_operations*100:.1f}%")
        print(f"Average rate: {total_operations/actual_duration:.1f} ops/sec")
        print(f"Peak memory: {peak_usage['peak_memory_percent']:.1f}%")
        print(f"Peak CPU: {peak_usage['peak_cpu_percent']:.1f}%")
        print(f"Max threads: {peak_usage['max_thread_count']}")

        return {
            'duration_minutes': actual_duration / 60,
            'total_operations': total_operations,
            'success_rate': total_successes / total_operations if total_operations > 0 else 0,
            'average_rate': total_operations / actual_duration,
            'peak_memory_percent': peak_usage['peak_memory_percent'],
            'peak_cpu_percent': peak_usage['peak_cpu_percent']
        }


if __name__ == "__main__":
    print("Running extended stress test...")
    results = run_extended_stress_test(duration_minutes=5, max_concurrent=200)
    print("Stress test completed:", results)
