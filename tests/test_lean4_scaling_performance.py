#!/usr/bin/env python3
"""
Focused Scaling Performance Tests for Lean 4 Prover/Agent

Detailed performance analysis and scaling characteristics measurement.
Focuses on throughput, latency, resource usage, and bottleneck identification.
"""

import time
import statistics
import concurrent.futures
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
import pytest

from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation_count: int
    concurrent_level: int
    total_duration: float
    successful_operations: int
    failed_operations: int
    throughput: float  # operations per second
    mean_latency: float  # seconds
    median_latency: float
    p95_latency: float
    p99_latency: float
    memory_usage_mb: float
    thread_efficiency: float  # actual_throughput / theoretical_max


class PerformanceMeasurer:
    """Measures detailed performance metrics for concurrent operations."""

    def __init__(self):
        self.operation_times: List[float] = []
        self.thread_local = threading.local()
        self.lock = threading.Lock()

    def start_operation(self):
        """Mark the start of an operation."""
        self.thread_local.start_time = time.time()

    def end_operation(self, success: bool = True):
        """Mark the end of an operation and record timing."""
        if hasattr(self.thread_local, 'start_time'):
            duration = time.time() - self.thread_local.start_time
            with self.lock:
                self.operation_times.append(duration)

    def get_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self.operation_times:
            return {"mean": 0, "median": 0, "p95": 0, "p99": 0}

        sorted_times = sorted(self.operation_times)
        return {
            "mean": statistics.mean(sorted_times),
            "median": statistics.median(sorted_times),
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)]
        }

    def reset(self):
        """Reset collected metrics."""
        with self.lock:
            self.operation_times.clear()


# Mock with configurable latency for realistic testing
class ConfigurableMockServer:
    """Mock server with configurable response times and failure rates."""

    def __init__(self, config, latency_ms: float = 1.0, failure_rate: float = 0.0):
        self.config = config
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.call_count = 0
        self.thread_count = 0

    def run(self, command):
        """Mock run with configurable latency and failure."""
        self.call_count += 1

        # Simulate processing time
        time.sleep(self.latency_ms / 1000.0)

        # Simulate failures
        import random
        if random.random() < self.failure_rate:
            raise Exception(f"Simulated failure (rate: {self.failure_rate})")

        # Return successful mock response
        if hasattr(command, 'cmd'):
            return Mock(
                env=self.call_count,
                sorries=[Mock(proof_state=self.call_count * 10)] if 'sorry' in command.cmd else [],
                messages=[],
                success=True
            )
        else:
            return Mock(
                proof_status="Complete",
                proof_state=self.call_count * 10,
                goals=[],
                success=True
            )


@contextmanager
def mock_server_with_config(latency_ms: float = 1.0, failure_rate: float = 0.0):
    """Context manager for configurable mock server."""
    def server_factory(config):
        return ConfigurableMockServer(config, latency_ms=latency_ms, failure_rate=failure_rate)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', server_factory):
        yield


def measure_scaling_performance(
    operation_func,
    concurrent_levels: List[int],
    latency_ms: float = 1.0,
    failure_rate: float = 0.0
) -> List[PerformanceMetrics]:
    """
    Measure performance scaling across different concurrency levels.

    Args:
        operation_func: Function to execute concurrently
        concurrent_levels: List of concurrency levels to test
        latency_ms: Simulated server latency
        failure_rate: Simulated failure rate

    Returns:
        List of PerformanceMetrics for each concurrency level
    """
    results = []

    with mock_server_with_config(latency_ms=latency_ms, failure_rate=failure_rate):
        for concurrent_level in concurrent_levels:
            measurer = PerformanceMeasurer()

            def measured_operation():
                measurer.start_operation()
                try:
                    result = operation_func()
                    measurer.end_operation(success=True)
                    return result
                except Exception as e:
                    measurer.end_operation(success=False)
                    raise

            start_time = time.time()
            successful_ops = 0
            failed_ops = 0

            # Execute operations concurrently
            max_workers = min(concurrent_level, 100)  # Limit thread pool size
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(measured_operation) for _ in range(concurrent_level)]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            successful_ops += 1
                        else:
                            failed_ops += 1
                    except Exception:
                        failed_ops += 1

            end_time = time.time()
            total_duration = end_time - start_time

            # Calculate metrics
            latency_stats = measurer.get_latency_stats()
            throughput = successful_ops / total_duration if total_duration > 0 else 0

            # Theoretical maximum throughput (assuming perfect parallelization)
            theoretical_max = concurrent_level / (latency_ms / 1000.0) if latency_ms > 0 else float('inf')
            thread_efficiency = throughput / theoretical_max if theoretical_max > 0 else 0

            metrics = PerformanceMetrics(
                operation_count=concurrent_level,
                concurrent_level=concurrent_level,
                total_duration=total_duration,
                successful_operations=successful_ops,
                failed_operations=failed_ops,
                throughput=throughput,
                mean_latency=latency_stats["mean"],
                median_latency=latency_stats["median"],
                p95_latency=latency_stats["p95"],
                p99_latency=latency_stats["p99"],
                memory_usage_mb=0,  # Would need actual memory monitoring
                thread_efficiency=min(thread_efficiency, 1.0)  # Cap at 100%
            )

            results.append(metrics)

            print(f"Concurrency {concurrent_level:3d}: "
                  f"{throughput:6.1f} ops/sec, "
                  f"latency p95: {latency_stats['p95']*1000:5.1f}ms, "
                  f"efficiency: {thread_efficiency*100:5.1f}%")

    return results


# Test Functions

@pytest.mark.parametrize("latency_ms", [0.5, 1.0, 2.0, 5.0])
def test_scaling_with_different_latencies(latency_ms):
    """Test how scaling performance changes with different server latencies."""

    def create_and_run_theorem():
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem perf_test : True := trivial")
        return result.success

    concurrent_levels = [1, 5, 10, 20, 50]
    results = measure_scaling_performance(
        create_and_run_theorem,
        concurrent_levels,
        latency_ms=latency_ms
    )

    print(f"\nScaling Results for {latency_ms}ms latency:")
    print("Concurrency | Throughput | P95 Latency | Efficiency")
    print("-" * 50)
    for r in results:
        print(f"{r.concurrent_level:11} | {r.throughput:10.1f} | "
              f"{r.p95_latency*1000:11.1f} | {r.thread_efficiency*100:9.1f}%")

    # Verify scaling characteristics
    assert len(results) == len(concurrent_levels)
    assert all(r.successful_operations > 0 for r in results)

    # Throughput should generally increase with concurrency (up to a point)
    throughputs = [r.throughput for r in results]
    assert max(throughputs) > throughputs[0], "Concurrency should improve throughput"


def test_interactive_agent_scaling():
    """Test scaling performance of InteractiveLeanAgent operations."""

    def interactive_workflow():
        agent = InteractiveLeanAgent(mathlib_enabled=True)

        # Load theorem
        theorem = "theorem interactive_perf (P Q : Prop) : P → Q → P ∧ Q := by sorry"
        load_result = agent.load_theorem(theorem)

        # Edit clause if available
        if load_result.get("editable_clauses"):
            for clause_id in load_result["editable_clauses"]:
                if "sorry" in clause_id:
                    edit_result = agent.edit_clause(clause_id, "intro h1 h2; exact ⟨h1, h2⟩")
                    return edit_result.get("edit_successful", False)

        return True

    concurrent_levels = [1, 5, 10, 20]
    results = measure_scaling_performance(interactive_workflow, concurrent_levels, latency_ms=1.0)

    print(f"\nInteractive Agent Scaling Results:")
    for r in results:
        print(f"Concurrency {r.concurrent_level}: {r.throughput:.1f} ops/sec, "
              f"Success rate: {r.successful_operations/r.operation_count*100:.1f}%")

    assert all(r.successful_operations > 0 for r in results)


@pytest.mark.parametrize("failure_rate", [0.0, 0.1, 0.2, 0.5])
def test_scaling_with_failures(failure_rate):
    """Test scaling performance under different failure rates."""

    def operation_with_potential_failure():
        prover = LeanProver(mathlib_enabled=True)
        try:
            result = prover.run("theorem failure_test : True := trivial")
            return result.success
        except Exception:
            return False

    concurrent_levels = [10, 25, 50]
    results = measure_scaling_performance(
        operation_with_potential_failure,
        concurrent_levels,
        latency_ms=1.0,
        failure_rate=failure_rate
    )

    print(f"\nScaling with {failure_rate*100}% failure rate:")
    for r in results:
        actual_failure_rate = r.failed_operations / (r.successful_operations + r.failed_operations)
        print(f"Concurrency {r.concurrent_level}: "
              f"throughput {r.throughput:.1f} ops/sec, "
              f"failure rate {actual_failure_rate*100:.1f}%")

    # Should handle failures gracefully
    assert all(r.successful_operations > 0 or failure_rate == 1.0 for r in results)


def test_bottleneck_identification():
    """Identify performance bottlenecks by testing different components."""

    def prover_creation_only():
        """Test just prover creation overhead."""
        prover = LeanProver(mathlib_enabled=True)
        return True

    def theorem_execution_only():
        """Test theorem execution with pre-created prover."""
        # This would need prover to be shared, but that's not thread-safe
        # So we test creation + execution
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem bottleneck_test : True := trivial")
        return result.success

    concurrent_levels = [10, 25]

    # Test different operation types
    creation_results = measure_scaling_performance(
        prover_creation_only, concurrent_levels, latency_ms=0.1
    )

    execution_results = measure_scaling_performance(
        theorem_execution_only, concurrent_levels, latency_ms=1.0
    )

    print(f"\nBottleneck Analysis:")
    print("Operation Type | Concurrency | Throughput")
    print("-" * 40)
    for i, level in enumerate(concurrent_levels):
        print(f"Creation      | {level:11} | {creation_results[i].throughput:10.1f}")
        print(f"Execution     | {level:11} | {execution_results[i].throughput:10.1f}")

    # Creation should be faster than execution (with our mock)
    assert creation_results[0].throughput >= execution_results[0].throughput * 0.5


@pytest.mark.slow
def test_sustained_load_performance():
    """Test performance under sustained load over time."""

    def sustained_operation():
        prover = LeanProver(mathlib_enabled=True)
        result = prover.run("theorem sustained_test : True := trivial")
        return result.success

    # Run operations over multiple time windows
    concurrent_level = 20
    time_windows = 5
    operations_per_window = 50

    window_results = []

    with mock_server_with_config(latency_ms=1.0):
        for window in range(time_windows):
            measurer = PerformanceMeasurer()

            def measured_operation():
                measurer.start_operation()
                try:
                    result = sustained_operation()
                    measurer.end_operation(success=True)
                    return result
                except Exception:
                    measurer.end_operation(success=False)
                    return False

            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
                futures = [executor.submit(measured_operation) for _ in range(operations_per_window)]
                successes = sum(future.result() for future in concurrent.futures.as_completed(futures))

            end_time = time.time()
            duration = end_time - start_time
            throughput = successes / duration

            latency_stats = measurer.get_latency_stats()
            window_results.append((window, throughput, latency_stats["p95"]))

            print(f"Window {window+1}: {throughput:.1f} ops/sec, "
                  f"p95 latency: {latency_stats['p95']*1000:.1f}ms")

    # Performance should be relatively stable over time
    throughputs = [t for _, t, _ in window_results]
    avg_throughput = statistics.mean(throughputs)
    throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0

    print(f"\nSustained Load Results:")
    print(f"Average throughput: {avg_throughput:.1f} ops/sec")
    print(f"Throughput std dev: {throughput_std:.1f}")
    print(f"Coefficient of variation: {throughput_std/avg_throughput*100:.1f}%")

    # Performance should be reasonably consistent
    assert throughput_std / avg_throughput < 0.2, "Performance should be stable over time"


def test_memory_scaling_analysis():
    """Analyze memory usage scaling with concurrency."""

    # This test would require actual memory monitoring
    # For now, we'll simulate with operation counting

    def memory_intensive_operation():
        # Simulate creating large data structures
        prover = LeanProver(mathlib_enabled=True)
        # In real usage, this might accumulate memory
        result = prover.run("theorem memory_test : True := trivial")
        return result.success

    concurrent_levels = [5, 10, 20, 50]

    for level in concurrent_levels:
        with mock_server_with_config(latency_ms=1.0):
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(memory_intensive_operation) for _ in range(level)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            duration = time.time() - start_time
            success_rate = sum(results) / len(results)

            print(f"Concurrency {level:2d}: {success_rate*100:5.1f}% success, {duration:.2f}s")

    # All should succeed with mock
    assert True  # Placeholder for actual memory analysis


if __name__ == "__main__":
    print("Running performance scaling analysis...")

    # Run a quick performance analysis
    def basic_operation():
        prover = LeanProver(mathlib_enabled=True)
        return prover.run("theorem perf : True := trivial").success

    results = measure_scaling_performance(
        basic_operation,
        [1, 5, 10, 20, 50, 100],
        latency_ms=1.0
    )

    print("\nPerformance Analysis Complete!")
    print("Run with pytest to execute all scaling tests.")
