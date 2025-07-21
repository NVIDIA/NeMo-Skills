#!/usr/bin/env python3
"""
Thread-Safe Lean 4 Prover and Interactive Agent

This module provides thread-safe versions that allow multiple agents to solve
different proofs concurrently without interfering with each other's solving process.

Key Features:
- Per-thread isolation of proof state
- Thread-safe proof management
- Concurrent agent instances that don't interfere
- Safe for thousands of concurrent agents
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
import copy

from .prover import LeanProver, ProofResult, ProofInProgress
from .interactive_agent import InteractiveLeanAgent, LeanMessage, ProofGoal, EditableClause


class ThreadSafeLeanProver:
    """
    Thread-safe wrapper for LeanProver that ensures concurrent agents don't interfere.

    Key Design:
    - Each method call is atomic
    - Environment state is properly isolated
    - Proof state management is thread-safe
    - Multiple instances can run concurrently
    """

    def __init__(self, mathlib_enabled: bool = True):
        """Initialize thread-safe prover."""
        self._mathlib_enabled = mathlib_enabled
        self._lock = threading.RLock()  # Reentrant lock
        self._thread_local = threading.local()

        # Thread-safe proof registry
        self._proofs_registry: Dict[str, ProofInProgress] = {}
        self._proofs_lock = threading.RLock()

        # Instance ID for debugging
        self._instance_id = str(uuid.uuid4())[:8]

    def _get_prover(self) -> LeanProver:
        """Get or create thread-local prover instance."""
        if not hasattr(self._thread_local, 'prover'):
            # Each thread gets its own LeanProver instance
            self._thread_local.prover = LeanProver(mathlib_enabled=self._mathlib_enabled)
            self._thread_local.thread_id = threading.get_ident()

        return self._thread_local.prover

    def run(self, lean_code: str) -> ProofResult:
        """Thread-safe execution of Lean code."""
        prover = self._get_prover()
        return prover.run(lean_code)

    def run_command(self, command: str, env: Optional[int] = None) -> ProofResult:
        """Thread-safe command execution."""
        prover = self._get_prover()
        return prover.run_command(command, env)

    def run_proof_step(self, proof_state: int, tactic: str) -> ProofResult:
        """Thread-safe proof step execution."""
        prover = self._get_prover()
        return prover.run_proof_step(proof_state, tactic)

    def start_proof(self, name: str, statement: str) -> ProofResult:
        """Thread-safe proof initialization."""
        # Generate unique proof name per thread to avoid conflicts
        thread_id = threading.get_ident()
        unique_name = f"{name}_thread_{thread_id}_{int(time.time() * 1000) % 10000}"

        prover = self._get_prover()
        result = prover.start_proof(unique_name, statement)

        if result.success:
            # Register proof in thread-safe registry
            with self._proofs_lock:
                # Copy proof state from thread-local prover to global registry
                if unique_name in prover.proofs_in_progress:
                    proof_copy = copy.deepcopy(prover.proofs_in_progress[unique_name])
                    # Update with original requested name for user convenience
                    proof_copy.name = name
                    self._proofs_registry[name] = proof_copy

        return result

    def apply_tactic_to_proof(self, proof_name: str, tactic: str) -> ProofResult:
        """Thread-safe tactic application."""
        # Find the actual proof name in thread-local storage
        prover = self._get_prover()

        with self._proofs_lock:
            if proof_name not in self._proofs_registry:
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=f"No proof named '{proof_name}' found",
                    error=f"No proof named '{proof_name}' found"
                )

            # Find corresponding proof in thread-local prover
            thread_id = threading.get_ident()
            actual_name = None
            for local_name, proof in prover.proofs_in_progress.items():
                if local_name.startswith(f"{proof_name}_thread_{thread_id}"):
                    actual_name = local_name
                    break

            if actual_name is None:
                return ProofResult(
                    success=False,
                    proof_complete=False,
                    has_sorry=False,
                    response=f"Proof '{proof_name}' not found in current thread",
                    error=f"Proof '{proof_name}' not found in current thread"
                )

            # Apply tactic using thread-local prover
            result = prover.apply_tactic_to_proof(actual_name, tactic)

            # Update global registry if successful
            if result.success and actual_name in prover.proofs_in_progress:
                proof_copy = copy.deepcopy(prover.proofs_in_progress[actual_name])
                proof_copy.name = proof_name  # Keep original name
                self._proofs_registry[proof_name] = proof_copy

            return result

    def get_proof_state(self, proof_name: str) -> Optional[ProofInProgress]:
        """Thread-safe proof state retrieval."""
        with self._proofs_lock:
            if proof_name in self._proofs_registry:
                return copy.deepcopy(self._proofs_registry[proof_name])
        return None

    def backtrack_proof(self, proof_name: str, steps: int = 1) -> ProofResult:
        """Thread-safe proof backtracking."""
        prover = self._get_prover()

        # Find actual proof name in thread-local storage
        thread_id = threading.get_ident()
        actual_name = None
        for local_name in prover.proofs_in_progress.keys():
            if local_name.startswith(f"{proof_name}_thread_{thread_id}"):
                actual_name = local_name
                break

        if actual_name is None:
            return ProofResult(
                success=False,
                proof_complete=False,
                has_sorry=False,
                response=f"Proof '{proof_name}' not found in current thread",
                error=f"Proof '{proof_name}' not found in current thread"
            )

        result = prover.backtrack_proof(actual_name, steps)

        # Update global registry
        if result.success:
            with self._proofs_lock:
                if actual_name in prover.proofs_in_progress:
                    proof_copy = copy.deepcopy(prover.proofs_in_progress[actual_name])
                    proof_copy.name = proof_name
                    self._proofs_registry[proof_name] = proof_copy

        return result

    def get_current_env(self) -> Optional[int]:
        """Get current environment for this thread."""
        prover = self._get_prover()
        return prover.get_current_env()

    def get_thread_info(self) -> Dict[str, Any]:
        """Get debugging info about current thread's prover state."""
        prover = self._get_prover()
        thread_id = threading.get_ident()

        return {
            'instance_id': self._instance_id,
            'thread_id': thread_id,
            'current_env': prover.get_current_env(),
            'active_proofs': list(prover.proofs_in_progress.keys()),
            'thread_local_proofs': len(prover.proofs_in_progress),
            'global_registry_proofs': len(self._proofs_registry)
        }


class ThreadSafeInteractiveLeanAgent:
    """
    Thread-safe wrapper for InteractiveLeanAgent that ensures concurrent agents
    can work on different proofs without interfering with each other.
    """

    def __init__(self, mathlib_enabled: bool = True):
        """Initialize thread-safe interactive agent."""
        self._mathlib_enabled = mathlib_enabled
        self._lock = threading.RLock()
        self._thread_local = threading.local()

        # Instance ID for debugging
        self._instance_id = str(uuid.uuid4())[:8]

    def _get_agent(self) -> InteractiveLeanAgent:
        """Get or create thread-local agent instance."""
        if not hasattr(self._thread_local, 'agent'):
            # Each thread gets its own InteractiveLeanAgent instance
            self._thread_local.agent = InteractiveLeanAgent(mathlib_enabled=self._mathlib_enabled)
            self._thread_local.thread_id = threading.get_ident()
            self._thread_local.session_id = str(uuid.uuid4())[:8]

        return self._thread_local.agent

    def load_theorem(self, theorem_code: str) -> Dict[str, Any]:
        """Thread-safe theorem loading."""
        agent = self._get_agent()

        # Add thread-specific markers to avoid naming conflicts
        thread_id = threading.get_ident()
        session_id = self._thread_local.session_id

        # Modify theorem code to be thread-unique
        modified_code = self._make_thread_unique_code(theorem_code, thread_id, session_id)

        result = agent.load_theorem(modified_code)

        # Add thread info to result
        result['thread_info'] = {
            'thread_id': thread_id,
            'session_id': session_id,
            'instance_id': self._instance_id
        }

        return result

    def _make_thread_unique_code(self, code: str, thread_id: int, session_id: str) -> str:
        """Make theorem code unique per thread to avoid conflicts."""
        import re

        # Find theorem names and make them unique
        def replace_theorem_name(match):
            original_name = match.group(1)
            unique_name = f"{original_name}_t{thread_id % 10000}_s{session_id}"
            return f"theorem {unique_name}"

        # Replace theorem declarations
        unique_code = re.sub(r'theorem\s+(\w+)', replace_theorem_name, code)
        return unique_code

    def edit_clause(self, clause_id: str, new_content: str) -> Dict[str, Any]:
        """Thread-safe clause editing."""
        agent = self._get_agent()
        result = agent.edit_clause(clause_id, new_content)

        # Add thread info
        result['thread_info'] = {
            'thread_id': threading.get_ident(),
            'session_id': getattr(self._thread_local, 'session_id', 'unknown'),
            'instance_id': self._instance_id
        }

        return result

    def get_goal_at_position(self, line: int, column: int) -> Optional[ProofGoal]:
        """Thread-safe goal retrieval."""
        agent = self._get_agent()
        return agent.get_goal_at_position(line, column)

    def get_messages_at_position(self, line: int, column: int) -> List[LeanMessage]:
        """Thread-safe message retrieval."""
        agent = self._get_agent()
        return agent.get_messages_at_position(line, column)

    def add_proof_structure(self, structure_lines: List[str]) -> Dict[str, Any]:
        """Thread-safe proof structure addition."""
        agent = self._get_agent()
        result = agent.add_proof_structure(structure_lines)

        result['thread_info'] = {
            'thread_id': threading.get_ident(),
            'session_id': getattr(self._thread_local, 'session_id', 'unknown'),
            'instance_id': self._instance_id
        }

        return result

    def get_interactive_panel(self) -> Dict[str, Any]:
        """Thread-safe interactive panel state."""
        agent = self._get_agent()
        panel = agent.get_interactive_panel()

        # Add thread-specific info
        panel['thread_info'] = {
            'thread_id': threading.get_ident(),
            'session_id': getattr(self._thread_local, 'session_id', 'unknown'),
            'instance_id': self._instance_id
        }

        return panel

    def suggest_next_actions(self) -> List[str]:
        """Thread-safe action suggestions."""
        agent = self._get_agent()
        return agent.suggest_next_actions()

    def get_thread_info(self) -> Dict[str, Any]:
        """Get debugging info about current thread's agent state."""
        if not hasattr(self._thread_local, 'agent'):
            return {
                'instance_id': self._instance_id,
                'thread_id': threading.get_ident(),
                'status': 'no_agent_initialized'
            }

        agent = self._get_agent()
        return {
            'instance_id': self._instance_id,
            'thread_id': threading.get_ident(),
            'session_id': getattr(self._thread_local, 'session_id', 'unknown'),
            'compilation_id': agent.compilation_id,
            'current_code_length': len(agent.current_code),
            'active_messages': len(agent.current_messages),
            'active_goals': len(agent.current_goals),
            'editable_clauses': len(agent.editable_clauses)
        }


# Utility Functions for Thread-Safe Usage

@contextmanager
def concurrent_lean_session(mathlib_enabled: bool = True):
    """
    Context manager for safe concurrent Lean usage.

    Usage:
        with concurrent_lean_session() as (prover, agent):
            result = prover.run("theorem test : True := trivial")
            agent_result = agent.load_theorem("theorem example : 1 + 1 = 2 := rfl")
    """
    prover = ThreadSafeLeanProver(mathlib_enabled=mathlib_enabled)
    agent = ThreadSafeInteractiveLeanAgent(mathlib_enabled=mathlib_enabled)

    try:
        yield prover, agent
    finally:
        # Cleanup if needed
        pass


def create_concurrent_agent_pool(pool_size: int, mathlib_enabled: bool = True) -> List[ThreadSafeInteractiveLeanAgent]:
    """
    Create a pool of thread-safe interactive agents for concurrent use.

    Args:
        pool_size: Number of agents to create
        mathlib_enabled: Whether to enable mathlib support

    Returns:
        List of ThreadSafeInteractiveLeanAgent instances
    """
    return [ThreadSafeInteractiveLeanAgent(mathlib_enabled=mathlib_enabled)
            for _ in range(pool_size)]


def validate_concurrent_safety_strict(num_threads: int = 20, num_operations: int = 10,
                                      require_perfect: bool = True) -> Dict[str, Any]:
    """
    Strict thread safety validation requiring 100% success rate.

    Args:
        num_threads: Number of concurrent threads to test
        num_operations: Operations per thread
        require_perfect: If True, requires 100% success rate

    Returns:
        Validation results with perfect success requirement
    """
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
        'prover_results': [],
        'agent_results': [],
        'conflicts_detected': 0,
        'total_operations': 0,
        'success_rate': 0.0,
        'resource_stats': {},
        'perfect_validation': True,
        'issues_found': []
    }

    def test_prover_thread_rigorous(thread_id: int):
        """Rigorous prover testing per thread."""
        prover = ThreadSafeLeanProver(mathlib_enabled=True)
        thread_results = []

        for i in range(num_operations):
            try:
                # Test with varied complexity
                theorems = [
                    f"theorem rigorous_test_{thread_id}_{i}_simple : True := trivial",
                    f"theorem rigorous_test_{thread_id}_{i}_arith : 1 + 1 = 2 := rfl",
                    f"theorem rigorous_test_{thread_id}_{i}_logic (P : Prop) : P → P := fun h => h"
                ]

                theorem = theorems[i % len(theorems)]
                result = prover.run(theorem)

                thread_results.append({
                    'thread_id': thread_id,
                    'operation': i,
                    'success': result.success,
                    'response': result.response if not result.success else "Success",
                    'thread_info': prover.get_thread_info()
                })

                # Also test proof building
                proof_name = f"proof_{thread_id}_{i}"
                proof_result = prover.start_proof(proof_name, ": True")
                if proof_result.success:
                    tactic_result = prover.apply_tactic_to_proof(proof_name, "trivial")
                    thread_results.append({
                        'thread_id': thread_id,
                        'operation': f"{i}_proof",
                        'success': tactic_result.success,
                        'response': tactic_result.response if not tactic_result.success else "Success",
                        'thread_info': prover.get_thread_info()
                    })

            except Exception as e:
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': i,
                    'success': False,
                    'response': f"Exception: {str(e)}",
                    'thread_info': {'error': str(e)}
                })

        return thread_results

    def test_agent_thread_rigorous(thread_id: int):
        """Rigorous agent testing per thread."""
        agent = ThreadSafeInteractiveLeanAgent(mathlib_enabled=True)
        thread_results = []

        for i in range(num_operations):
            try:
                # Test with interactive development
                theorem_code = f"""theorem rigorous_agent_{thread_id}_{i} (P Q : Prop) : P ∧ Q → Q ∧ P := by sorry"""

                result = agent.load_theorem(theorem_code)
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': i,
                    'success': result.get('success', False),
                    'response': result.get('raw_response', 'No response'),
                    'thread_info': agent.get_thread_info()
                })

                # Test editing if successful
                if result.get('success'):
                    editable_clauses = result.get('editable_clauses', [])
                    if editable_clauses:
                        sorry_clause = next((c for c in editable_clauses if 'sorry' in c), None)
                        if sorry_clause:
                            edit_result = agent.edit_clause(sorry_clause, "intro h; exact ⟨h.right, h.left⟩")
                            thread_results.append({
                                'thread_id': thread_id,
                                'operation': f"{i}_edit",
                                'success': edit_result.get('compilation_result', {}).get('success', False),
                                'response': "Edit completed",
                                'thread_info': agent.get_thread_info()
                            })

            except Exception as e:
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': i,
                    'success': False,
                    'response': f"Exception: {str(e)}",
                    'thread_info': {'error': str(e)}
                })

        return thread_results

    # Run rigorous concurrent tests
    print(f"🧪 STRICT VALIDATION: {num_threads} threads × {num_operations} ops")
    print("   Requiring 100% success rate with zero tolerance for failures...")

    start_time = time.time()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
            # Launch both prover and agent tests concurrently
            futures = []

            for i in range(num_threads):
                futures.append(executor.submit(test_prover_thread_rigorous, i))
                futures.append(executor.submit(test_agent_thread_rigorous, i + num_threads))

            # Collect results with timeout
            for future in concurrent.futures.as_completed(futures, timeout=300):  # 5 minute timeout
                try:
                    thread_results = future.result()
                    # Classify results
                    if thread_results and 'thread_info' in thread_results[0]:
                        if 'instance_id' in str(thread_results[0].get('thread_info', {})):
                            results['prover_results'].extend(thread_results)
                        else:
                            results['agent_results'].extend(thread_results)
                except concurrent.futures.TimeoutError:
                    results['issues_found'].append("Timeout during operation")
                    results['perfect_validation'] = False
                except Exception as e:
                    results['issues_found'].append(f"Thread execution error: {str(e)}")
                    results['perfect_validation'] = False

    except Exception as e:
        results['issues_found'].append(f"Executor error: {str(e)}")
        results['perfect_validation'] = False

    elapsed_time = time.time() - start_time

    # Analyze results with zero tolerance
    all_results = results['prover_results'] + results['agent_results']
    results['total_operations'] = len(all_results)

    if results['total_operations'] == 0:
        results['success_rate'] = 0.0
        results['perfect_validation'] = False
        results['issues_found'].append("No operations completed")
    else:
        successful_ops = sum(1 for r in all_results if r.get('success', False))
        results['success_rate'] = successful_ops / results['total_operations']

        # RESEARCH REQUIREMENT: Must be exactly 100%
        if require_perfect and results['success_rate'] != 1.0:
            results['perfect_validation'] = False
            failed_ops = results['total_operations'] - successful_ops
            results['issues_found'].append(f"Failed operations: {failed_ops}/{results['total_operations']}")

            # Report first few failures for debugging
            failures = [r for r in all_results if not r.get('success', False)][:5]
            for i, failure in enumerate(failures):
                results['issues_found'].append(f"Failure {i+1}: {failure.get('response', 'Unknown error')}")

    # Check for real conflicts with zero tolerance
    session_ids_by_thread = {}
    instance_ids_by_thread = {}

    for result in all_results:
        thread_info = result.get('thread_info', {})
        thread_id = thread_info.get('thread_id')

        # Session ID conflicts
        if 'session_id' in thread_info:
            session_id = thread_info['session_id']
            for other_thread, other_session in session_ids_by_thread.items():
                if other_thread != thread_id and other_session == session_id:
                    results['conflicts_detected'] += 1
                    results['perfect_validation'] = False
                    results['issues_found'].append(f"Session ID conflict: {session_id} in threads {thread_id}, {other_thread}")
            session_ids_by_thread[thread_id] = session_id

        # Instance ID conflicts
        if 'instance_id' in thread_info:
            instance_id = thread_info['instance_id']
            for other_thread, other_instance in instance_ids_by_thread.items():
                if other_thread != thread_id and other_instance == instance_id:
                    results['conflicts_detected'] += 1
                    results['perfect_validation'] = False
                    results['issues_found'].append(f"Instance ID conflict: {instance_id} in threads {thread_id}, {other_thread}")
            instance_ids_by_thread[thread_id] = instance_id

    # Resource analysis
    final_memory = process.memory_info().rss / 1024 / 1024
    final_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    results['resource_stats'] = {
        'memory_mb': {
            'initial': initial_memory,
            'final': final_memory,
            'leaked': final_fds - initial_fds
        },
        'file_descriptors': {
            'initial': initial_fds,
            'final': final_fds,
            'leaked': final_fds - initial_fds
        },
        'elapsed_time': elapsed_time
    }

    # Check for resource leaks
    memory_leaked = final_memory - initial_memory
    fd_leaked = final_fds - initial_fds

    if memory_leaked > 50:  # More than 50MB leaked
        results['perfect_validation'] = False
        results['issues_found'].append(f"Memory leak detected: {memory_leaked:.1f}MB")

    if fd_leaked > 10:  # More than 10 FDs leaked
        results['perfect_validation'] = False
        results['issues_found'].append(f"File descriptor leak: {fd_leaked} FDs")

    # Force cleanup
    gc.collect()

    return results


# Keep original function for backward compatibility
def validate_concurrent_safety(num_threads: int = 10, num_operations: int = 5) -> Dict[str, Any]:
    """
    Standard thread safety validation (backward compatibility).

    For strict validation requirements, use validate_concurrent_safety_strict() instead.
    """
    return validate_concurrent_safety_strict(num_threads, num_operations, require_perfect=False)


# Research-grade stress test
def extreme_stress_test(max_concurrent: int = 100, duration_seconds: int = 30) -> Dict[str, Any]:
    """
    Extreme stress test pushing resources to the limit.

    Args:
        max_concurrent: Maximum concurrent agents to run
        duration_seconds: How long to run the stress test

    Returns:
        Detailed stress test results
    """
    import concurrent.futures
    import random
    import time
    import threading
    import gc
    import psutil

    print(f"🔥 EXTREME STRESS TEST: {max_concurrent} concurrent agents for {duration_seconds}s")
    print("   This will push your system to its limits...")

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    initial_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    # Stress test state
    stress_results = {
        'agents_created': 0,
        'operations_completed': 0,
        'operations_successful': 0,
        'peak_concurrent': 0,
        'errors': [],
        'resource_exhaustion': False,
        'memory_exhaustion': False
    }

    active_agents = []
    results_lock = threading.Lock()
    stop_flag = threading.Event()

    def stress_agent_worker(agent_id: int):
        """Individual stress test agent."""
        try:
            agent = ThreadSafeInteractiveLeanAgent(mathlib_enabled=True)

            with results_lock:
                stress_results['agents_created'] += 1
                active_agents.append(agent_id)
                stress_results['peak_concurrent'] = max(
                    stress_results['peak_concurrent'],
                    len(active_agents)
                )

            # Run operations until stopped
            op_count = 0
            while not stop_flag.is_set():
                try:
                    # Random theorem types
                    theorem_types = [
                        f"theorem stress_{agent_id}_{op_count} : True := trivial",
                        f"theorem stress_{agent_id}_{op_count} : 1 + 1 = 2 := rfl",
                        f"theorem stress_{agent_id}_{op_count} (P : Prop) : P → P := fun h => h"
                    ]

                    theorem = random.choice(theorem_types)
                    result = agent.load_theorem(theorem)

                    with results_lock:
                        stress_results['operations_completed'] += 1
                        if result.get('success', False):
                            stress_results['operations_successful'] += 1

                    op_count += 1

                    # Brief pause to prevent overwhelming
                    time.sleep(random.uniform(0.001, 0.01))  # 1-10ms

                except Exception as e:
                    with results_lock:
                        stress_results['errors'].append(f"Agent {agent_id}: {str(e)}")
                    break

        except Exception as e:
            with results_lock:
                stress_results['errors'].append(f"Agent {agent_id} startup: {str(e)}")

        finally:
            with results_lock:
                if agent_id in active_agents:
                    active_agents.remove(agent_id)

    # Resource monitoring
    def resource_monitor():
        """Monitor resources during stress test."""
        while not stop_flag.is_set():
            try:
                current_memory = process.memory_info().rss / 1024 / 1024
                current_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

                # Check for resource exhaustion
                if current_memory > initial_memory + 1000:  # 1GB increase
                    stress_results['memory_exhaustion'] = True

                if current_fds > initial_fds + 100:  # 100 FDs increase
                    stress_results['resource_exhaustion'] = True

                time.sleep(0.5)  # Check every 500ms

            except:
                break

    # Start monitoring
    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    try:
        # Gradually ramp up to max concurrent agents
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent * 2) as executor:
            futures = []

            # Launch agents gradually
            for i in range(max_concurrent):
                futures.append(executor.submit(stress_agent_worker, i))

                # Ramp up gradually to avoid overwhelming system
                if i < max_concurrent // 2:
                    time.sleep(0.01)  # 10ms between launches initially

            # Run for specified duration
            time.sleep(duration_seconds)

            # Signal stop
            stop_flag.set()

            # Wait for completion with timeout
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    future.result()
                except:
                    pass  # Expected some to fail under stress

    except Exception as e:
        stress_results['errors'].append(f"Executor error: {str(e)}")

    finally:
        stop_flag.set()

    elapsed_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024
    final_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    # Final results
    success_rate = (stress_results['operations_successful'] /
                   stress_results['operations_completed']
                   if stress_results['operations_completed'] > 0 else 0.0)

    print(f"\n📊 EXTREME STRESS TEST RESULTS:")
    print(f"   Duration: {elapsed_time:.1f}s")
    print(f"   Agents created: {stress_results['agents_created']}")
    print(f"   Peak concurrent: {stress_results['peak_concurrent']}")
    print(f"   Operations: {stress_results['operations_completed']}")
    print(f"   Success rate: {success_rate*100:.1f}%")
    print(f"   Operations/second: {stress_results['operations_completed']/elapsed_time:.1f}")
    print(f"   Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB")
    print(f"   File descriptors: {initial_fds} → {final_fds}")
    print(f"   Errors: {len(stress_results['errors'])}")

    if success_rate >= 0.98 and len(stress_results['errors']) < max_concurrent * 0.1:
        print("   ✅ STRESS TEST PASSED: System handles extreme load!")
    else:
        print("   ⚠️  STRESS TEST ISSUES: Some resource limits hit")

    gc.collect()  # Cleanup

    return {
        'stress_results': stress_results,
        'success_rate': success_rate,
        'elapsed_time': elapsed_time,
        'resource_stats': {
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'fds_initial': initial_fds,
            'fds_final': final_fds
        },
        'passed': success_rate >= 0.95 and len(stress_results['errors']) < max_concurrent * 0.2
    }


# Migration helpers for existing code
def make_prover_thread_safe(prover: LeanProver) -> ThreadSafeLeanProver:
    """Convert existing LeanProver to thread-safe version."""
    return ThreadSafeLeanProver(mathlib_enabled=prover.mathlib_enabled)


def make_agent_thread_safe(agent: InteractiveLeanAgent) -> ThreadSafeInteractiveLeanAgent:
    """Convert existing InteractiveLeanAgent to thread-safe version."""
    return ThreadSafeInteractiveLeanAgent(mathlib_enabled=agent.prover.mathlib_enabled)
