# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytest-compatible session affinity tests for the multi-worker sandbox server.
Tests verify that session state persists across requests and that session routing works correctly.
"""

import json
import random
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
import requests

BASE_URL = "http://localhost:6000"


class SessionAffinityTester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.lock = threading.Lock()

    def execute_code(self, code, session_id, timeout=30, language="ipython"):
        """Execute code with session and return result with timing info"""
        start_time = time.time()
        try:
            payload = {
                "generated_code": code,
                "timeout": timeout,
                "language": language,
            }

            headers = {
                "Content-Type": "application/json",
            }
            if session_id:
                headers["X-Session-ID"] = session_id

            # HTTP timeout should be longer than execution timeout + buffer for IPython session creation
            http_timeout = max(timeout + 10, 30)  # IPython sessions start much faster than Jupyter kernels
            response = requests.post(f"{self.base_url}/execute", json=payload, headers=headers, timeout=http_timeout)
            end_time = time.time()

            if response.status_code != 200:
                return {
                    "process_status": "http_error",
                    "stdout": "",
                    "stderr": f"HTTP {response.status_code}: {response.text[:500]}",
                    "response_time": end_time - start_time,
                    "timestamp": end_time,
                }

            try:
                result = response.json()
            except json.JSONDecodeError as e:
                return {
                    "process_status": "json_error",
                    "stdout": "",
                    "stderr": f"JSON decode error: {e}. Response status: {response.status_code}, Content-Type: {response.headers.get('content-type', 'unknown')}, Body: '{response.text[:500]}'",
                    "response_time": end_time - start_time,
                    "timestamp": end_time,
                }

            result["response_time"] = end_time - start_time
            result["timestamp"] = end_time

            return result

        except Exception as e:
            return {
                "process_status": "request_error",
                "stdout": "",
                "stderr": f"Request failed: {str(e)}",
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
            }

    def test_session_persistence(self, session_id, num_operations=5):
        """Test that session state persists across multiple operations"""
        operations = []
        session_suffix = str(hash(session_id))[-4:]  # Get unique suffix

        # 1. Initialize session with a variable
        init_code = f"""session_var_{session_suffix} = 'initialized_{session_suffix}'
print(f'Session {session_suffix} initialized')"""

        result = self.execute_code(init_code, session_id)
        operations.append(("init", result))

        if result["process_status"] != "completed":
            return operations, False, "Session initialization failed"

        # 2. Add imports that should persist
        import_code = """import random
import math
print('Imports added')"""
        result = self.execute_code(import_code, session_id)
        operations.append(("import", result))

        if result["process_status"] != "completed":
            return operations, False, "Import failed"

        # 3. Define a function that should persist
        func_code = f"""def session_func_{session_suffix}(x):
    return x * 2 + len(session_var_{session_suffix})

print(f'Function defined for session {session_suffix}')"""

        result = self.execute_code(func_code, session_id)
        operations.append(("func_def", result))

        if result["process_status"] != "completed":
            return operations, False, "Function definition failed"

        # 4. Test operations using persistent state
        for i in range(num_operations - 3):
            # Small random delay to potentially hit different workers
            time.sleep(random.uniform(0.01, 0.05))

            operation_num = i + 1
            test_code = f"""# Test that all previous state exists
try:
    # Test variable exists
    print(f'Variable: {{session_var_{session_suffix}}}')

    # Test imports work
    rand_val = random.randint(1, 100)
    sqrt_val = math.sqrt(rand_val)
    print(f'Random: {{rand_val}}, Sqrt: {{sqrt_val:.2f}}')

    # Test function works
    func_result = session_func_{session_suffix}(rand_val)
    print(f'Function result: {{func_result}}')

    # Update state for next iteration
    session_var_{session_suffix} += f'_op{operation_num}'

    print(f'Operation {operation_num} completed successfully')

except NameError as e:
    print(f'SESSION MISS DETECTED: {{e}}')
    raise e
except Exception as e:
    print(f'Other error: {{e}}')
    raise e"""

            result = self.execute_code(test_code, session_id)
            operations.append((f"op_{operation_num}", result))

            # Check for session miss
            if (
                "SESSION MISS DETECTED" in result.get("stdout", "")
                or "NameError" in result.get("stderr", "")
                or result.get("process_status") != "completed"
            ):
                return operations, False, f"Session miss in operation {operation_num}"

        return operations, True, "All operations successful"

    def get_worker_info(self, session_id=None, language="ipython"):
        """Get worker info by executing code that reveals worker details"""
        code = """
import os
worker_port = os.environ.get('LISTEN_PORT', 'unknown')
worker_num = os.environ.get('WORKER_NUM', 'unknown')
process_id = os.getpid()
print(f"WORKER_INFO: port={worker_port}, num={worker_num}, pid={process_id}")
"""

        headers = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        result = self.execute_code(code, session_id, language=language)  # Will use headers via execute_code

        # For session requests, we get JSON response with stdout
        if result.get("process_status") == "completed":
            stdout = result.get("stdout", "")
            if "WORKER_INFO:" in stdout:
                info_line = [line for line in stdout.split("\n") if "WORKER_INFO:" in line][0]
                # Parse: port=6001, num=1, pid=12345
                parts = info_line.replace("WORKER_INFO: ", "").split(", ")
                info = {}
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        info[key] = value
                return info

        # For non-session requests (session_id=None), we get HTML response as json_error
        elif result.get("process_status") == "json_error":
            # Extract worker info from HTML body in stderr message
            stderr = result.get("stderr", "")
            if "Body:" in stderr and "WORKER_INFO:" in stderr:
                # Extract the body content: "Body: 'WORKER_INFO: port=6047, num=47, pid=4075\n'"
                body_start = stderr.find("Body: '") + len("Body: '")
                body_end = stderr.find("'", body_start)
                if body_start > len("Body: '") - 1 and body_end > body_start:
                    body_content = stderr[body_start:body_end]
                    if "WORKER_INFO:" in body_content:
                        info_line = [line for line in body_content.split("\n") if "WORKER_INFO:" in line][0]
                        # Parse: port=6001, num=1, pid=12345
                        parts = info_line.replace("WORKER_INFO: ", "").split(", ")
                        info = {}
                        for part in parts:
                            if "=" in part:
                                key, value = part.split("=", 1)
                                info[key] = value
                        return info
        return None


@pytest.fixture
def tester():
    """Fixture providing a SessionAffinityTester instance"""
    return SessionAffinityTester()


class TestSessionAffinity:
    """Test class for session affinity functionality"""

    def test_server_health(self):
        """Test that the server is responding to health checks"""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Server not available: HTTP {response.status_code}")

    def test_basic_session_persistence(self, tester):
        """Test that a single session maintains state across requests"""
        session_id = f"test_basic_{int(time.time())}"
        operations, success, message = tester.test_session_persistence(session_id, num_operations=3)

        assert success, f"Session persistence failed: {message}"
        assert len(operations) == 3

        # Verify all operations completed successfully
        for op_name, result in operations:
            assert result["process_status"] == "completed", (
                f"Operation {op_name} failed: {result.get('stderr', 'Unknown error')}"
            )

    @pytest.mark.parametrize("num_operations", [3, 5, 8])
    def test_session_persistence_various_lengths(self, tester, num_operations):
        """Test session persistence with different numbers of operations"""
        session_id = f"test_length_{num_operations}_{int(time.time())}"
        operations, success, message = tester.test_session_persistence(session_id, num_operations)

        assert success, f"Session persistence failed with {num_operations} operations: {message}"
        assert len(operations) == num_operations

    def test_multiple_concurrent_sessions(self, tester):
        """Test that multiple concurrent sessions don't interfere with each other"""
        num_sessions = 10
        operations_per_session = 4

        def test_single_session(session_num):
            session_id = f"test_concurrent_{session_num}_{int(time.time())}"
            return tester.test_session_persistence(session_id, operations_per_session)

        # Run sessions concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(test_single_session, i) for i in range(num_sessions)]
            results = [future.result() for future in as_completed(futures)]

        # Analyze results
        successful_sessions = 0
        failed_sessions = []

        for i, (operations, success, message) in enumerate(results):
            if success:
                successful_sessions += 1
            else:
                failed_sessions.append((i, message))

        # Assert that most sessions succeeded (allow for some flakiness under load)
        success_rate = successful_sessions / num_sessions
        assert success_rate >= 1.0, f"Success rate too low: {success_rate:.1%}. Failed sessions: {failed_sessions}"

    def test_session_affinity_routing(self, tester):
        """Test that the same session_id consistently routes to the same worker"""
        session_id = f"test_routing_{int(time.time())}"

        workers_hit = set()
        num_requests = 10

        for i in range(num_requests):
            worker_info = tester.get_worker_info(session_id)
            if worker_info and "port" in worker_info:
                workers_hit.add(worker_info["port"])
            time.sleep(0.1)  # Small delay between requests

        # All requests with same session_id should hit the same worker
        assert len(workers_hit) == 1, (
            f"Session affinity broken: session hit {len(workers_hit)} different workers: {workers_hit}"
        )

    def test_session_persistence_large_payload(self, tester):
        """Test session persistence with large payloads exceeding nginx body buffer"""
        LARGE_PAYLOAD_SIZE = 1024 * 1024 + 1024  # 257KB to exceed 128KB buffer
        session_id = f"test_large_payload_{uuid.uuid4()}"

        # Generate large code string (>128KB)
        large_var = "x" * (LARGE_PAYLOAD_SIZE // 2)  # Roughly half size for string
        init_code = f"""
large_var = '{large_var}'
print('Large variable initialized')
"""

        result = tester.execute_code(init_code, session_id)
        assert result["process_status"] == "completed", "Failed to initialize large variable"
        assert "Large variable initialized" in result["stdout"]

        # Second request: use the large variable and add more data
        large_addition = "y" * (LARGE_PAYLOAD_SIZE // 2)
        use_code = f"""
try:
    combined = large_var + '{large_addition}'
    print(f'Combined length: {{len(combined)}}')
    print('Large payload test successful')
except NameError:
    print('SESSION MISS: large_var not found')
"""

        result = tester.execute_code(use_code, session_id)
        assert result["process_status"] == "completed", "Failed to use large variable"
        assert "Large payload test successful" in result["stdout"]
        assert "SESSION MISS" not in result["stdout"]

    def test_multiple_large_payloads_concurrent(self, tester):
        """Test concurrent sessions with large payloads"""
        LARGE_PAYLOAD_SIZE = 1024 * 1024 + 1024  # 257KB to exceed 128KB buffer
        num_sessions = 3

        def test_large_session(i):
            session_id = f"concurrent_large_{i}_{uuid.uuid4()}"

            large_var = "z" * LARGE_PAYLOAD_SIZE
            code1 = f"large_var = '{large_var}'; print('Initialized')"
            res1 = tester.execute_code(code1, session_id)

            code2 = "print(len(large_var))"
            res2 = tester.execute_code(code2, session_id)

            return (
                res1["process_status"] == "completed"
                and res2["process_status"] == "completed"
                and str(LARGE_PAYLOAD_SIZE) in res2["stdout"]
            )

        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            results = list(executor.map(test_large_session, range(num_sessions)))

        assert all(results), f"{results.count(False)} concurrent large payload sessions failed"

    def test_different_sessions_can_hit_different_workers(self, tester):
        """Test that different session_ids can potentially hit different workers"""
        num_sessions = 10
        workers_hit = set()

        for i in range(num_sessions):
            session_id = f"test_distribution_{i}_{int(time.time())}"
            worker_info = tester.get_worker_info(session_id)
            if worker_info and "port" in worker_info:
                workers_hit.add(worker_info["port"])

        # We should see some distribution across workers (unless there's only 1 worker)
        # This test might pass even with 1 worker, which is okay
        assert len(workers_hit) >= 1, "No workers responded to requests"

    def test_load_balancing_without_session_id(self, tester):
        """Test that requests without session_id distribute across workers"""
        workers_hit = set()
        num_requests = 20

        for i in range(num_requests):
            worker_info = tester.get_worker_info(session_id=None, language="python")  # No session_id
            if worker_info and "port" in worker_info:
                workers_hit.add(worker_info["port"])
            time.sleep(0.05)

        # Without session affinity, we should see some distribution
        # (unless there's only 1 worker)
        assert len(workers_hit) >= 1, "No workers responded to requests"

    @pytest.mark.parametrize(
        "session_config",
        [
            {"sessions": 5, "ops": 3, "workers": 3, "name": "Light Load"},
            {"sessions": 10, "ops": 4, "workers": 5, "name": "Medium Load"},
            {"sessions": 20, "ops": 3, "workers": 8, "name": "Heavy Load"},
        ],
    )
    def test_session_affinity_under_load(self, tester, session_config):
        """Test session affinity under various load conditions"""
        num_sessions = session_config["sessions"]
        operations_per_session = session_config["ops"]
        max_workers = session_config["workers"]

        def test_single_session(session_num):
            session_id = f"test_load_{session_config['name']}_{session_num}_{int(time.time())}"
            return tester.test_session_persistence(session_id, operations_per_session)

        # Run load test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(test_single_session, i) for i in range(num_sessions)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()

        # Analyze results
        successful_sessions = sum(1 for _, success, _ in results if success)
        total_operations = num_sessions * operations_per_session
        session_misses = 0

        for operations, success, message in results:
            if not success and "Session miss" in message:
                session_misses += 1

        # Assert no session misses (critical for session affinity)
        assert session_misses == 0, (
            f"Session affinity failure: {session_misses} session misses detected in {session_config['name']}"
        )

        # Assert reasonable success rate (allowing for some other types of failures)
        success_rate = successful_sessions / num_sessions
        assert success_rate >= 1.0, f"Success rate too low for {session_config['name']}: {success_rate:.1%}"

        print(
            f"{session_config['name']}: {successful_sessions}/{num_sessions} sessions successful in {end_time - start_time:.1f}s"
        )

    def test_session_cleanup_endpoint(self, tester):
        """Test that session management endpoints work correctly"""
        session_id = f"test_cleanup_{int(time.time())}"

        # Create a session by executing some code
        result = tester.execute_code("test_var = 'cleanup_test'", session_id)
        assert result["process_status"] == "completed"

        # Verify session exists by using the variable
        result = tester.execute_code("print(test_var)", session_id)
        assert result["process_status"] == "completed"
        assert "cleanup_test" in result.get("stdout", "")

        # Delete the session - include session_id in header for proper routing
        delete_response = requests.delete(f"{BASE_URL}/sessions/{session_id}", headers={"X-Session-ID": session_id})
        assert delete_response.status_code == 200

        # Verify session is gone (this should fail with NameError)
        result = tester.execute_code("print(test_var)", session_id)
        # Note: After deletion, a new session will be created, so test_var won't exist
        assert "NameError" in result.get("stderr", "") or result["process_status"] == "error"

    def test_session_list_endpoint(self):
        """Test that we can list active sessions"""
        # Create a few sessions
        session_ids = []
        for i in range(3):
            session_id = f"test_list_{i}_{int(time.time())}"
            session_ids.append(session_id)

            # Execute code to create the session
            requests.post(
                f"{BASE_URL}/execute",
                json={"generated_code": f"list_test_var_{i} = {i}", "timeout": 10, "language": "ipython"},
                headers={"X-Session-ID": session_id},
            )

        # List sessions
        list_response = requests.get(f"{BASE_URL}/sessions")
        assert list_response.status_code == 200

        session_data = list_response.json()
        assert "sessions" in session_data
        assert "backend" in session_data
        assert session_data["backend"] == "ipython"

        # Verify our sessions are in the list
        active_sessions = session_data["sessions"]
        for session_id in session_ids:
            # Session might not be in list if it was cleaned up, so this is not a hard requirement
            # Just verify the structure is correct if sessions exist
            if session_id in active_sessions:
                session_info = active_sessions[session_id]
                assert "backend" in session_info
                assert "created" in session_info
                assert "last_used" in session_info
                assert "alive" in session_info

    def test_load_balancing(self):
        """Test that requests without session ID are distributed across multiple workers"""
        num_requests = 2560
        workers = []
        for _ in range(num_requests):
            resp = requests.get(f"{BASE_URL}/health")
            assert resp.status_code == 200
            workers.append(resp.json()["worker"])
        unique_workers = set(workers)
        assert len(unique_workers) > 1, f"All requests went to the same worker: {unique_workers}"
        counts = Counter(workers)

        # Check for and identify the full range of active workers
        int_keys = sorted(int(k) for k in counts if k != "unknown")
        min_key = min(int_keys)
        max_key = max(int_keys)
        expected_keys = list(range(min_key, max_key + 1))
        assert len(int_keys) == len(expected_keys), (
            f"Gaps in worker numbers: missing {set(expected_keys) - set(int_keys)}"
        )

        # Chi-Squared Goodness-of-Fit Test
        num_workers = len(expected_keys)
        observed_counts = np.array(list(counts.values()))
        expected_count = num_requests / num_workers

        # The chi-squared statistic measures the deviation from the expected distribution
        chi2_stat = np.sum((observed_counts - expected_count) ** 2 / expected_count)

        degrees_of_freedom = num_workers - 1
        z_score = 2.0
        critical_value = 0.5 * (z_score + np.sqrt(2 * degrees_of_freedom - 1)) ** 2
        assert chi2_stat <= critical_value, (
            f"Uneven distribution of jobs across uwsgi workers. "
            f"Observed counts: {sorted(counts.items())} (min={np.min(observed_counts)}, "
            f"max={np.max(observed_counts)}, avg={expected_count:.1f}). "
            f"Chi-squared statistic ({chi2_stat:.2f}) > Critical Value ({critical_value:.2f}) "
            f"for {degrees_of_freedom} degrees of freedom"
        )


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
