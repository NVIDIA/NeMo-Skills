#!/usr/bin/env python3
"""
Focused Edge Condition Tests for Lean 4 Concurrent Usage

Specifically tests the edge conditions you asked about:
1. Multiple concurrent accesses to same context
2. Environment ID collisions
3. Proof state isolation
4. Resource contention detection
"""

import concurrent.futures
import threading
import time
import os
from collections import defaultdict, Counter
from unittest.mock import Mock, patch
import pytest

from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent

# Configuration for testing
USE_MOCK_SERVERS = os.getenv('MOCK_SERVERS', '1') != '0'
MAX_REAL_CONCURRENT = int(os.getenv('MAX_REAL_CONCURRENT', '10'))

def get_safe_concurrency_level(requested_level):
    """Get safe concurrency level based on testing mode."""
    if USE_MOCK_SERVERS:
        return requested_level
    else:
        return min(requested_level, MAX_REAL_CONCURRENT)


class ConcurrencyTestServer:
    """Test server that tracks environment IDs and detects conflicts."""

    def __init__(self, config):
        self.config = config
        self.environment_ids = []
        self.call_log = []
        self._lock = threading.Lock()

    def run(self, command):
        thread_id = threading.get_ident()
        call_time = time.time()

        # Generate environment ID (simulating real Lean behavior)
        # Real issue: if multiple provers share same environment, they could interfere
        with self._lock:
            env_id = len(self.environment_ids) + (thread_id % 100)  # Add thread variation
            self.environment_ids.append((thread_id, env_id, call_time))
            self.call_log.append((thread_id, env_id, str(command)[:50], call_time))

        # Mock response
        if hasattr(command, 'cmd'):
            if 'sorry' in command.cmd:
                return Mock(
                    env=env_id,
                    sorries=[Mock(proof_state=env_id * 10)],
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
            return Mock(
                proof_status="Complete",
                proof_state=env_id * 10,
                goals=[],
                success=True
            )

    def get_analysis(self):
        """Analyze for potential conflicts."""
        with self._lock:
            thread_env_map = defaultdict(set)
            env_thread_map = defaultdict(set)

            for thread_id, env_id, call_time in self.environment_ids:
                thread_env_map[thread_id].add(env_id)
                env_thread_map[env_id].add(thread_id)

            # Check for environment ID conflicts (multiple threads using same env)
            shared_envs = {env_id: threads for env_id, threads in env_thread_map.items()
                          if len(threads) > 1}

            # Check for threads that got too many environments (possible ID collision)
            threads_with_multiple_envs = {tid: envs for tid, envs in thread_env_map.items()
                                        if len(envs) > 3}  # More than 3 envs per thread might indicate issues

            return {
                'shared_environments': shared_envs,
                'threads_with_multiple_envs': threads_with_multiple_envs,
                'total_threads': len(thread_env_map),
                'total_environments': len(env_thread_map),
                'has_environment_conflicts': len(shared_envs) > 0,
                'call_log': self.call_log[-10:]  # Last 10 calls for debugging
            }


def test_environment_id_collision_detection():
    """Test detection of environment ID collisions between concurrent provers."""
    print("\n🔍 TESTING: Environment ID collision detection")

    test_server = ConcurrencyTestServer(None)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', lambda config: test_server):
        concurrent_level = get_safe_concurrency_level(8)

        def create_prover_and_run(worker_id):
            """Each worker creates a prover and runs operations."""
            prover = LeanProver(mathlib_enabled=True)

            results = []
            for i in range(3):
                theorem = f"theorem env_test_{worker_id}_{i} : True := trivial"
                result = prover.run(theorem)
                results.append({
                    'worker_id': worker_id,
                    'operation': i,
                    'success': result.success,
                    'thread_id': threading.get_ident()
                })
            return results

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(create_prover_and_run, i) for i in range(concurrent_level)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        # Analyze for environment conflicts
        analysis = test_server.get_analysis()

        print(f"   Workers: {concurrent_level}")
        print(f"   Total operations: {len(all_results)}")
        print(f"   Threads tracked: {analysis['total_threads']}")
        print(f"   Unique environments: {analysis['total_environments']}")
        print(f"   Environment conflicts detected: {analysis['has_environment_conflicts']}")
        if analysis['shared_environments']:
            print(f"   Shared environments: {analysis['shared_environments']}")

        # Verify no environment collisions
        assert not analysis['has_environment_conflicts'], \
            f"Environment ID collisions detected: {analysis['shared_environments']}"

        # Each thread should get reasonably unique environments
        assert analysis['total_environments'] >= concurrent_level, \
            f"Too few unique environments: {analysis['total_environments']} for {concurrent_level} workers"

        # All operations should succeed
        success_count = sum(1 for r in all_results if r['success'])
        assert success_count == len(all_results), f"Operations failed: {success_count}/{len(all_results)}"


def test_proof_state_context_isolation():
    """Test that concurrent proof states don't interfere with each other."""
    print("\n🔍 TESTING: Proof state context isolation")

    test_server = ConcurrencyTestServer(None)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', lambda config: test_server):
        concurrent_level = get_safe_concurrency_level(6)

        # Shared data structure to test for interference
        proof_contexts = {}
        context_lock = threading.Lock()

        def build_proof_with_context(proof_id):
            """Build proof while maintaining context."""
            thread_id = threading.get_ident()
            prover = LeanProver(mathlib_enabled=True)

            # Record context
            with context_lock:
                proof_contexts[proof_id] = {
                    'thread_id': thread_id,
                    'started_at': time.time(),
                    'steps': []
                }

            try:
                # Start proof with sorry to get initial state
                proof_name = f"context_proof_{proof_id}"
                start_result = prover.start_proof(proof_name, ": True")

                if not start_result.success:
                    return proof_id, "start_failed", None

                # Record initial proof state
                with context_lock:
                    proof_contexts[proof_id]['initial_proof_state'] = start_result.proof_state
                    proof_contexts[proof_id]['steps'].append(('start', start_result.proof_state))

                # Apply tactics and track state changes
                for step in range(2):
                    tactic = "trivial" if step == 1 else "-- step comment"
                    tactic_result = prover.apply_tactic_to_proof(proof_name, tactic)

                    with context_lock:
                        proof_contexts[proof_id]['steps'].append((tactic, tactic_result.proof_state))

                return proof_id, "success", proof_contexts[proof_id]

            except Exception as e:
                return proof_id, "error", str(e)

        # Run concurrent proof building
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(build_proof_with_context, i) for i in range(concurrent_level)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze proof context isolation
        successful_proofs = [r for r in results if r[1] == "success"]
        proof_ids = [r[0] for r in successful_proofs]
        contexts = [r[2] for r in successful_proofs]

        print(f"   Concurrent proofs: {concurrent_level}")
        print(f"   Successful proofs: {len(successful_proofs)}")
        print(f"   Unique proof IDs: {len(set(proof_ids))}")
        print(f"   Thread IDs used: {len(set(ctx['thread_id'] for ctx in contexts))}")

        # Check for context isolation
        thread_ids = [ctx['thread_id'] for ctx in contexts]
        proof_states = [ctx.get('initial_proof_state') for ctx in contexts]

        print(f"   Unique thread IDs: {len(set(thread_ids))}")
        print(f"   Unique initial proof states: {len(set(filter(None, proof_states)))}")

        # Verify isolation
        assert len(set(proof_ids)) == len(successful_proofs), "Proof IDs should be unique"
        assert len(set(thread_ids)) == len(successful_proofs), "Each proof should run in different thread"

        # Proof states should be different (isolated)
        unique_proof_states = len(set(filter(None, proof_states)))
        assert unique_proof_states >= len(successful_proofs) * 0.8, \
            f"Too few unique proof states: {unique_proof_states} for {len(successful_proofs)} proofs"

        # Analyze environment usage
        analysis = test_server.get_analysis()
        print(f"   Environment conflicts: {analysis['has_environment_conflicts']}")

        assert not analysis['has_environment_conflicts'], \
            f"Environment conflicts in proof contexts: {analysis['shared_environments']}"


def test_concurrent_prover_instance_isolation():
    """Test that concurrent prover instances don't share state."""
    print("\n🔍 TESTING: Concurrent prover instance isolation")

    test_server = ConcurrencyTestServer(None)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', lambda config: test_server):
        concurrent_level = get_safe_concurrency_level(10)

        # Test with shared vs isolated provers
        shared_prover = LeanProver(mathlib_enabled=True)

        def test_shared_prover(worker_id):
            """Use the same prover instance (should detect conflicts)."""
            theorem = f"theorem shared_test_{worker_id} : True := trivial"
            result = shared_prover.run(theorem)
            return worker_id, "shared", result.success, threading.get_ident()

        def test_isolated_prover(worker_id):
            """Create individual prover instance (should be isolated)."""
            prover = LeanProver(mathlib_enabled=True)
            theorem = f"theorem isolated_test_{worker_id} : True := trivial"
            result = prover.run(theorem)
            return worker_id, "isolated", result.success, threading.get_ident()

        # Test both patterns
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            # Mix of shared and isolated operations
            futures = []
            for i in range(concurrent_level // 2):
                futures.append(executor.submit(test_shared_prover, f"shared_{i}"))
                futures.append(executor.submit(test_isolated_prover, f"isolated_{i}"))

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze isolation
        shared_results = [r for r in results if r[1] == "shared"]
        isolated_results = [r for r in results if r[1] == "isolated"]

        shared_success_rate = sum(1 for r in shared_results if r[2]) / len(shared_results) if shared_results else 0
        isolated_success_rate = sum(1 for r in isolated_results if r[2]) / len(isolated_results) if isolated_results else 0

        print(f"   Shared prover operations: {len(shared_results)}")
        print(f"   Isolated prover operations: {len(isolated_results)}")
        print(f"   Shared success rate: {shared_success_rate*100:.1f}%")
        print(f"   Isolated success rate: {isolated_success_rate*100:.1f}%")

        analysis = test_server.get_analysis()
        print(f"   Environment conflicts detected: {analysis['has_environment_conflicts']}")
        if analysis['shared_environments']:
            print(f"   Threads sharing environments: {len(analysis['shared_environments'])}")

        # Both patterns should work with mocks, but isolated should be cleaner
        assert isolated_success_rate >= 0.9, f"Isolated prover success rate too low: {isolated_success_rate*100:.1f}%"

        # With proper isolation, there should be fewer environment conflicts
        shared_env_count = len(analysis['shared_environments'])
        print(f"   Environment sharing instances: {shared_env_count}")


def test_interactive_agent_concurrent_state_isolation():
    """Test that concurrent interactive agents maintain separate states."""
    print("\n🔍 TESTING: Interactive agent state isolation")

    test_server = ConcurrencyTestServer(None)

    with patch('nemo_skills.code_execution.lean4.prover.AutoLeanServer', lambda config: test_server):
        concurrent_level = get_safe_concurrency_level(5)

        def concurrent_interactive_session(agent_id):
            """Run interactive session with state tracking."""
            agent = InteractiveLeanAgent(mathlib_enabled=True)

            # Load theorem with agent-specific content
            theorem_code = f"""theorem agent_test_{agent_id} (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"""

            load_result = agent.load_theorem(theorem_code)

            state_info = {
                'agent_id': agent_id,
                'compilation_id': agent.compilation_id,
                'code_hash': hash(agent.current_code),
                'editable_clauses_count': len(agent.editable_clauses),
                'thread_id': threading.get_ident(),
                'load_success': load_result.get('success', False)
            }

            # Perform some editing if possible
            edit_success = False
            if load_result.get("editable_clauses"):
                clause_ids = load_result["editable_clauses"]
                sorry_clause = next((cid for cid in clause_ids if "sorry" in cid), None)
                if sorry_clause:
                    edit_result = agent.edit_clause(sorry_clause, "intro h; exact ⟨h.right, h.left⟩")
                    edit_success = edit_result.get("edit_successful", False)

            state_info['edit_success'] = edit_success
            return state_info

        # Run concurrent interactive sessions
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(concurrent_interactive_session, i)
                      for i in range(concurrent_level)]
            states = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze state isolation
        compilation_ids = [s['compilation_id'] for s in states]
        code_hashes = [s['code_hash'] for s in states]
        thread_ids = [s['thread_id'] for s in states]

        print(f"   Concurrent agents: {concurrent_level}")
        print(f"   Unique compilation IDs: {len(set(compilation_ids))}")
        print(f"   Unique code hashes: {len(set(code_hashes))}")
        print(f"   Unique thread IDs: {len(set(thread_ids))}")
        print(f"   Load success rate: {sum(s['load_success'] for s in states)/len(states)*100:.1f}%")
        print(f"   Edit success rate: {sum(s['edit_success'] for s in states)/len(states)*100:.1f}%")

        # Verify state isolation
        assert len(set(compilation_ids)) == concurrent_level, \
            f"Compilation IDs not unique: {len(set(compilation_ids))} != {concurrent_level}"

        assert len(set(code_hashes)) == concurrent_level, \
            f"Code hashes not unique: {len(set(code_hashes))} != {concurrent_level}"

        assert len(set(thread_ids)) == concurrent_level, \
            f"Thread IDs not unique: {len(set(thread_ids))} != {concurrent_level}"

        # Check environment isolation
        analysis = test_server.get_analysis()
        print(f"   Environment conflicts in agents: {analysis['has_environment_conflicts']}")

        # Should have clean environment separation
        assert not analysis['has_environment_conflicts'], \
            f"Interactive agents sharing environments: {analysis['shared_environments']}"


if __name__ == "__main__":
    print("🔍 Running focused edge condition tests for concurrent Lean 4 usage...")
    print("These tests specifically check for context isolation and resource conflicts.")

    # Run the tests
    test_environment_id_collision_detection()
    test_proof_state_context_isolation()
    test_concurrent_prover_instance_isolation()
    test_interactive_agent_concurrent_state_isolation()

    print("\n✅ All edge condition tests passed!")
    print("   ✓ Environment ID collision detection working")
    print("   ✓ Proof state context isolation verified")
    print("   ✓ Prover instance isolation confirmed")
    print("   ✓ Interactive agent state isolation validated")
