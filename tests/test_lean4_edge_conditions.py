#!/usr/bin/env python3
"""
Edge Condition Tests for Lean 4 Concurrent Usage

Tests for critical edge conditions in concurrent access:
- Context/Environment ID isolation
- Proof state corruption detection
- Shared resource conflicts
- Race condition validation
- Memory corruption detection
- File system isolation
"""

import concurrent.futures
import threading
import time
import os
from collections import defaultdict, Counter
from typing import Set, Dict, List, Any
from unittest.mock import Mock, patch
import pytest

from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent
from tests.test_lean4_concurrent import USE_MOCK_SERVERS, MAX_REAL_CONCURRENT, lean_server_context, get_safe_concurrency_level


class ThreadSafeTracker:
    """Thread-safe tracker for detecting race conditions and conflicts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'environment_ids': [],
            'proof_state_ids': [],
            'thread_contexts': {},
            'access_times': [],
            'errors': []
        }

    def record_environment_access(self, thread_id: int, env_id: int):
        """Record environment ID access by thread."""
        with self._lock:
            self._data['environment_ids'].append((thread_id, env_id, time.time()))

    def record_proof_state_access(self, thread_id: int, proof_state_id: int):
        """Record proof state ID access by thread."""
        with self._lock:
            self._data['proof_state_ids'].append((thread_id, proof_state_id, time.time()))

    def record_thread_context(self, thread_id: int, context_data: Dict):
        """Record thread-specific context data."""
        with self._lock:
            self._data['thread_contexts'][thread_id] = context_data

    def record_error(self, thread_id: int, error: str):
        """Record thread-specific error."""
        with self._lock:
            self._data['errors'].append((thread_id, error, time.time()))

    def get_analysis(self) -> Dict[str, Any]:
        """Analyze collected data for race conditions and conflicts."""
        with self._lock:
            analysis = {
                'environment_conflicts': self._analyze_environment_conflicts(),
                'proof_state_conflicts': self._analyze_proof_state_conflicts(),
                'thread_isolation': self._analyze_thread_isolation(),
                'temporal_conflicts': self._analyze_temporal_conflicts(),
                'error_summary': self._analyze_errors()
            }
        return analysis

    def _analyze_environment_conflicts(self):
        """Detect environment ID conflicts between threads."""
        env_by_thread = defaultdict(set)
        env_access_count = Counter()

        for thread_id, env_id, timestamp in self._data['environment_ids']:
            env_by_thread[thread_id].add(env_id)
            env_access_count[env_id] += 1

        # Find environment IDs used by multiple threads
        shared_envs = {env_id: count for env_id, count in env_access_count.items() if count > 1}

        return {
            'shared_environment_ids': shared_envs,
            'threads_per_env': len(env_by_thread),
            'unique_envs_per_thread': {tid: len(envs) for tid, envs in env_by_thread.items()},
            'has_conflicts': len(shared_envs) > 0
        }

    def _analyze_proof_state_conflicts(self):
        """Detect proof state conflicts between threads."""
        proof_by_thread = defaultdict(set)
        proof_access_count = Counter()

        for thread_id, proof_id, timestamp in self._data['proof_state_ids']:
            proof_by_thread[thread_id].add(proof_id)
            proof_access_count[proof_id] += 1

        shared_proofs = {proof_id: count for proof_id, count in proof_access_count.items() if count > 1}

        return {
            'shared_proof_state_ids': shared_proofs,
            'threads_with_proofs': len(proof_by_thread),
            'has_conflicts': len(shared_proofs) > 0
        }

    def _analyze_thread_isolation(self):
        """Check if threads have properly isolated contexts."""
        contexts = self._data['thread_contexts']

        # Check for real context data conflicts (sharing same objects/references)
        real_conflicts = []

        # Group by context keys and values to find problematic patterns
        key_value_threads = defaultdict(list)

        for thread_id, context in contexts.items():
            for key, value in context.items():
                # Skip expected identical values that don't indicate problems
                if key in ['call_count']:  # call_count being the same is expected if threads do same work
                    continue

                key_value_threads[(key, value)].append(thread_id)

        # Look for potentially problematic shared values
        for (key, value), thread_list in key_value_threads.items():
            if len(thread_list) > 1:
                # Check if this indicates a real problem
                if key in ['thread_id', 'environment_id']:  # These should be unique per thread
                    real_conflicts.append({
                        'key': key,
                        'value': value,
                        'threads': thread_list,
                        'type': 'shared_identity',
                        'severity': 'high'
                    })
                elif isinstance(value, int) and key.endswith('_id'):  # Other ID fields
                    real_conflicts.append({
                        'key': key,
                        'value': value,
                        'threads': thread_list,
                        'type': 'shared_resource_id',
                        'severity': 'medium'
                    })

        return {
            'total_threads': len(contexts),
            'real_conflicts': real_conflicts,
            'has_isolation_issues': len(real_conflicts) > 0,
            'context_summary': {
                key: len(set(ctx.get(key) for ctx in contexts.values() if key in ctx))
                for key in set().union(*(ctx.keys() for ctx in contexts.values()))
            }
        }

    def _analyze_temporal_conflicts(self):
        """Analyze timing-based conflicts (operations happening too close together)."""
        all_accesses = []
        all_accesses.extend([('env', data) for data in self._data['environment_ids']])
        all_accesses.extend([('proof', data) for data in self._data['proof_state_ids']])

        # Sort by timestamp
        all_accesses.sort(key=lambda x: x[1][2])  # Sort by timestamp

        # Find operations that happened within dangerous time windows
        dangerous_overlaps = []
        time_threshold = 0.001  # 1ms - operations this close could have race conditions

        for i in range(len(all_accesses) - 1):
            current_time = all_accesses[i][1][2]
            next_time = all_accesses[i + 1][1][2]

            if next_time - current_time < time_threshold:
                dangerous_overlaps.append({
                    'operations': [all_accesses[i], all_accesses[i + 1]],
                    'time_gap': next_time - current_time
                })

        return {
            'total_operations': len(all_accesses),
            'dangerous_overlaps': dangerous_overlaps,
            'has_timing_conflicts': len(dangerous_overlaps) > 0
        }

    def _analyze_errors(self):
        """Summarize errors by type and thread."""
        error_by_thread = defaultdict(list)
        error_types = Counter()

        for thread_id, error, timestamp in self._data['errors']:
            error_by_thread[thread_id].append(error)
            error_types[error] += 1

        return {
            'total_errors': len(self._data['errors']),
            'errors_by_thread': dict(error_by_thread),
            'error_types': dict(error_types)
        }


class AdvancedMockServer:
    """Advanced mock server that tracks context isolation and detects conflicts."""

    def __init__(self, config, tracker: ThreadSafeTracker):
        self.config = config
        self.tracker = tracker
        self.call_count = 0
        self.thread_call_counts = defaultdict(int)
        self.environment_map = {}  # Map call_count to env_id
        self._lock = threading.Lock()

    def run(self, command):
        thread_id = threading.get_ident()

        with self._lock:
            self.call_count += 1
            self.thread_call_counts[thread_id] += 1
            current_call = self.call_count

        # Simulate some processing time
        time.sleep(0.001)

        # Generate environment ID based on call pattern
        # This simulates how real Lean servers might assign environment IDs
        env_id = current_call + (thread_id % 1000)  # Add thread variation

        # Track environment access
        self.tracker.record_environment_access(thread_id, env_id)

        # Record thread context
        self.tracker.record_thread_context(thread_id, {
            'thread_id': thread_id,
            'call_count': self.thread_call_counts[thread_id],
            'environment_id': env_id
        })

        # Mock response
        if hasattr(command, 'cmd'):
            if 'sorry' in command.cmd:
                proof_state_id = env_id * 10 + current_call % 10
                self.tracker.record_proof_state_access(thread_id, proof_state_id)
                return Mock(
                    env=env_id,
                    sorries=[Mock(proof_state=proof_state_id)],
                    messages=[],
                    success=True
                )
            else:
                return Mock(
                    env=env_id,
                    sorries=[],
                    messages=[],
                    success=True
                )
        else:
            # ProofStep
            proof_state_id = env_id * 10 + current_call % 10
            self.tracker.record_proof_state_access(thread_id, proof_state_id)
            return Mock(
                proof_status="Complete",
                proof_state=proof_state_id,
                goals=[],
                success=True
            )


# Edge Condition Tests

def test_environment_id_isolation():
    """Test that concurrent provers get isolated environment IDs."""
    tracker = ThreadSafeTracker()

    def create_advanced_mock(config):
        return AdvancedMockServer(config, tracker)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', create_advanced_mock):
        concurrent_level = get_safe_concurrency_level(10)

        def isolated_prover_operation(worker_id):
            """Operation that should have isolated context."""
            prover = LeanProver(mathlib_enabled=True)

            # Perform multiple operations to test isolation
            results = []
            for i in range(3):
                theorem = f"theorem isolation_test_{worker_id}_{i} : True := trivial"
                result = prover.run(theorem)
                results.append(result.success)

            return worker_id, results

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(isolated_prover_operation, i)
                      for i in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze for conflicts
        analysis = tracker.get_analysis()

        print(f"\n🔍 ENVIRONMENT ISOLATION ANALYSIS:")
        print(f"   Threads tested: {concurrent_level}")
        print(f"   Environment conflicts: {analysis['environment_conflicts']['has_conflicts']}")
        print(f"   Shared environment IDs: {len(analysis['environment_conflicts']['shared_environment_ids'])}")
        print(f"   Thread isolation issues: {analysis['thread_isolation']['has_isolation_issues']}")
        print(f"   Timing conflicts: {analysis['temporal_conflicts']['has_timing_conflicts']}")

        # Assertions for proper isolation
        assert not analysis['environment_conflicts']['has_conflicts'], \
            f"Environment ID conflicts detected: {analysis['environment_conflicts']['shared_environment_ids']}"

        assert not analysis['thread_isolation']['has_isolation_issues'], \
            f"Thread isolation issues: {analysis['thread_isolation']['real_conflicts']}"

        # Each thread should have created its own prover instance
        worker_ids = [worker_id for worker_id, _ in results]
        assert len(set(worker_ids)) == concurrent_level, "All workers should be unique"

        # All operations should succeed
        all_successes = [all(result_list) for _, result_list in results]
        assert all(all_successes), "All operations should succeed"


def test_proof_state_isolation():
    """Test that concurrent proof states don't interfere with each other."""
    tracker = ThreadSafeTracker()

    def create_advanced_mock(config):
        return AdvancedMockServer(config, tracker)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', create_advanced_mock):
        concurrent_level = get_safe_concurrency_level(8)

        def concurrent_proof_building(proof_id):
            """Build proofs concurrently to test state isolation."""
            prover = LeanProver(mathlib_enabled=True)
            proof_name = f"concurrent_proof_{proof_id}"

            try:
                # Start proof
                start_result = prover.start_proof(proof_name, ": True")
                if not start_result.success:
                    return proof_id, "start_failed", None

                # Apply tactics
                for step in range(3):
                    tactic_result = prover.apply_tactic_to_proof(proof_name, f"-- step {step}")
                    if not tactic_result.success:
                        return proof_id, "tactic_failed", step

                # Complete proof
                complete_result = prover.apply_tactic_to_proof(proof_name, "trivial")

                return proof_id, "success", complete_result.success

            except Exception as e:
                tracker.record_error(threading.get_ident(), str(e))
                return proof_id, "error", str(e)

        # Run concurrent proof building
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(concurrent_proof_building, i)
                      for i in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze proof state conflicts
        analysis = tracker.get_analysis()

        print(f"\n🔍 PROOF STATE ISOLATION ANALYSIS:")
        print(f"   Proofs tested: {concurrent_level}")
        print(f"   Proof state conflicts: {analysis['proof_state_conflicts']['has_conflicts']}")
        print(f"   Shared proof states: {len(analysis['proof_state_conflicts']['shared_proof_state_ids'])}")
        print(f"   Errors: {analysis['error_summary']['total_errors']}")

        # Verify results
        proof_ids = [pid for pid, _, _ in results]
        statuses = [status for _, status, _ in results]

        # Each proof should have unique ID
        assert len(set(proof_ids)) == concurrent_level, "All proof IDs should be unique"

        # No proof state conflicts
        assert not analysis['proof_state_conflicts']['has_conflicts'], \
            f"Proof state conflicts detected: {analysis['proof_state_conflicts']['shared_proof_state_ids']}"

        # Most operations should succeed
        success_count = sum(1 for status in statuses if status == "success")
        assert success_count >= concurrent_level * 0.8, \
            f"Too many failures: {success_count}/{concurrent_level}"


def test_concurrent_resource_contention():
    """Test for resource contention and corruption under high concurrent load."""
    tracker = ThreadSafeTracker()

    def create_advanced_mock(config):
        return AdvancedMockServer(config, tracker)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', create_advanced_mock):
        concurrent_level = get_safe_concurrency_level(15)
        operations_per_worker = 5

        # Shared resource simulation
        shared_counter = {'value': 0}
        shared_lock = threading.Lock()

        def resource_contention_test(worker_id):
            """Test that simulates resource contention."""
            prover = LeanProver(mathlib_enabled=True)
            local_results = []

            for op in range(operations_per_worker):
                try:
                    # Simulate accessing shared resource
                    with shared_lock:
                        shared_counter['value'] += 1
                        local_snapshot = shared_counter['value']

                    # Perform Lean operation
                    theorem = f"theorem resource_{worker_id}_{op} : True := trivial"
                    result = prover.run(theorem)

                    # Verify no corruption
                    with shared_lock:
                        current_value = shared_counter['value']

                    if current_value < local_snapshot:
                        tracker.record_error(
                            threading.get_ident(),
                            f"Resource corruption: {current_value} < {local_snapshot}"
                        )

                    local_results.append({
                        'operation': op,
                        'lean_success': result.success,
                        'resource_snapshot': local_snapshot,
                        'final_resource_value': current_value
                    })

                except Exception as e:
                    tracker.record_error(threading.get_ident(), str(e))
                    local_results.append({'operation': op, 'error': str(e)})

            return worker_id, local_results

        # Execute under high contention
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(resource_contention_test, i)
                      for i in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()

        # Analyze contention results
        analysis = tracker.get_analysis()

        total_operations = sum(len(result_list) for _, result_list in results)
        successful_operations = 0
        resource_errors = 0

        for worker_id, result_list in results:
            for result in result_list:
                if result.get('lean_success'):
                    successful_operations += 1
                if 'error' in result and 'corruption' in result['error']:
                    resource_errors += 1

        print(f"\n🔍 RESOURCE CONTENTION ANALYSIS:")
        print(f"   Workers: {concurrent_level}")
        print(f"   Total operations: {total_operations}")
        print(f"   Successful operations: {successful_operations}")
        print(f"   Resource errors: {resource_errors}")
        print(f"   Duration: {end_time - start_time:.2f}s")
        print(f"   Final shared counter: {shared_counter['value']}")
        print(f"   Expected counter: {concurrent_level * operations_per_worker}")
        print(f"   Timing conflicts: {len(analysis['temporal_conflicts']['dangerous_overlaps'])}")

        # Verify no resource corruption
        assert resource_errors == 0, f"Resource corruption detected: {resource_errors} errors"

        # Verify counter integrity
        expected_counter = concurrent_level * operations_per_worker
        assert shared_counter['value'] == expected_counter, \
            f"Counter corruption: {shared_counter['value']} != {expected_counter}"

        # Most Lean operations should succeed
        success_rate = successful_operations / total_operations
        assert success_rate >= 0.9, f"Success rate too low: {success_rate*100:.1f}%"


def test_interactive_agent_state_isolation():
    """Test that concurrent interactive agents maintain isolated state."""
    tracker = ThreadSafeTracker()

    def create_advanced_mock(config):
        return AdvancedMockServer(config, tracker)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', create_advanced_mock):
        concurrent_level = get_safe_concurrency_level(6)

        def interactive_isolation_test(agent_id):
            """Test interactive agent state isolation."""
            agent = InteractiveLeanAgent(mathlib_enabled=True)

            # Load theorem with unique content
            theorem_code = f"""theorem interactive_isolation_{agent_id} (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"""

            load_result = agent.load_theorem(theorem_code)

            # Record agent state
            state_info = {
                'agent_id': agent_id,
                'compilation_id': agent.compilation_id,
                'code_length': len(agent.current_code),
                'editable_clauses': list(agent.editable_clauses.keys()),
                'messages_count': len(agent.current_messages)
            }

            tracker.record_thread_context(threading.get_ident(), state_info)

            # Perform editing operations
            edit_results = []
            if load_result.get("editable_clauses"):
                for clause_id in load_result["editable_clauses"]:
                    if "sorry" in clause_id:
                        edit_result = agent.edit_clause(clause_id, "intro h; exact ⟨h.right, h.left⟩")
                        edit_results.append(edit_result.get("edit_successful", False))

            return agent_id, state_info, edit_results

        # Run concurrent interactive sessions
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(interactive_isolation_test, i)
                      for i in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze state isolation
        analysis = tracker.get_analysis()

        # Verify each agent has unique state
        compilation_ids = []
        code_lengths = []

        for agent_id, state_info, edit_results in results:
            compilation_ids.append(state_info['compilation_id'])
            code_lengths.append(state_info['code_length'])

        print(f"\n🔍 INTERACTIVE AGENT ISOLATION ANALYSIS:")
        print(f"   Agents tested: {concurrent_level}")
        print(f"   Unique compilation IDs: {len(set(compilation_ids))}")
        print(f"   Context isolation issues: {analysis['thread_isolation']['has_isolation_issues']}")
        print(f"   Average edit success: {sum(len(edits) for _, _, edits in results) / len(results):.1f}")

        # Each agent should have unique compilation ID
        assert len(set(compilation_ids)) == concurrent_level, \
            "Interactive agents should have unique compilation IDs"

        # No context sharing between agents
        assert not analysis['thread_isolation']['has_isolation_issues'], \
            f"Agent state isolation failed: {analysis['thread_isolation']['real_conflicts']}"

        # Most edit operations should succeed
        total_edits = sum(len(edits) for _, _, edits in results)
        successful_edits = sum(sum(edits) for _, _, edits in results)
        if total_edits > 0:
            edit_success_rate = successful_edits / total_edits
            assert edit_success_rate >= 0.8, f"Edit success rate too low: {edit_success_rate*100:.1f}%"


@pytest.mark.skipif(not USE_MOCK_SERVERS, reason="Edge condition test requires mock servers for safety")
def test_extreme_concurrency_edge_cases():
    """Test edge cases under extreme concurrency to find race conditions."""
    tracker = ThreadSafeTracker()

    def create_advanced_mock(config):
        return AdvancedMockServer(config, tracker)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', create_advanced_mock):
        extreme_level = 50  # High concurrency only with mocks

        def extreme_edge_test(worker_id):
            """Extreme test designed to trigger race conditions."""
            results = []

            # Rapid-fire operations
            for i in range(10):
                prover = LeanProver(mathlib_enabled=True)

                # Multiple rapid operations on same prover
                for j in range(3):
                    theorem = f"theorem extreme_{worker_id}_{i}_{j} : True := trivial"
                    result = prover.run(theorem)
                    results.append(result.success)

                # Immediate cleanup and recreation (tests initialization race conditions)
                del prover

            return worker_id, results

        # Execute extreme concurrency
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=extreme_level) as executor:
            futures = [executor.submit(extreme_edge_test, i) for i in range(extreme_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()

        # Analyze extreme conditions
        analysis = tracker.get_analysis()

        total_ops = sum(len(result_list) for _, result_list in results)
        successful_ops = sum(sum(result_list) for _, result_list in results)

        print(f"\n🔥 EXTREME CONCURRENCY EDGE CASE ANALYSIS:")
        print(f"   Concurrent workers: {extreme_level}")
        print(f"   Total operations: {total_ops}")
        print(f"   Success rate: {successful_ops/total_ops*100:.1f}%")
        print(f"   Duration: {end_time - start_time:.2f}s")
        print(f"   Operations per second: {total_ops/(end_time - start_time):.1f}")
        print(f"   Environment conflicts: {analysis['environment_conflicts']['has_conflicts']}")
        print(f"   Timing conflicts: {len(analysis['temporal_conflicts']['dangerous_overlaps'])}")
        print(f"   Total errors: {analysis['error_summary']['total_errors']}")

        # Under extreme load, we still expect high success rates
        success_rate = successful_ops / total_ops
        assert success_rate >= 0.95, f"Success rate under extreme load too low: {success_rate*100:.1f}%"

        # Should not have environment conflicts even under extreme load
        assert not analysis['environment_conflicts']['has_conflicts'], \
            "Environment conflicts detected under extreme concurrency"


if __name__ == "__main__":
    print("Running edge condition tests...")
    pytest.main([__file__, "-v"])
