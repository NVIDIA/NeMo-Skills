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


import logging
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import time
import threading
from io import StringIO

from flask import Flask, request
import jupyter_client
import queue

app = Flask(__name__)

# Configurable session backend
SESSION_BACKEND = os.environ.get('SESSION_BACKEND', 'jupyter')  # jupyter, exec, ipython

@app.route("/debug/worker", methods=["GET", "POST"])
def debug_worker():
    """Debug endpoint to see which worker is handling the request"""
    import os

    # Check if this is a POST request with session_id
    session_from_json = None
    if request.method == 'POST' and request.is_json:
        data = request.get_json(silent=True)
        session_from_json = data.get('session_id') if data else None

    worker_info = {
        "worker_port": os.environ.get('LISTEN_PORT', 'unknown'),
        "worker_num": os.environ.get('WORKER_NUM', 'unknown'),
        "process_id": os.getpid(),
        "request_method": request.method,
        "session_id_from_json": session_from_json,
        "session_id_from_header": request.headers.get('X-Session-ID', 'none'),
        "nginx_extracted_session": request.headers.get('X-Session-ID', 'none'),
        "original_session_header": request.headers.get('X-Original-Session-Header', 'none'),
        "session_backend": SESSION_BACKEND
    }
    return worker_info

# ============================================================================
# JUPYTER BACKEND (Original - slow startup, full features)
# ============================================================================

# Global dictionary to store Jupyter kernels by session_id
kernels = {}
kernel_lock = threading.Lock()
KERNEL_TIMEOUT = 3600  # 1 hour timeout for kernels

# Configurable kernel startup parameters
KERNEL_READY_TIMEOUT = int(os.environ.get('KERNEL_READY_TIMEOUT', '60'))  # Default 60 seconds
KERNEL_STARTUP_RETRIES = int(os.environ.get('KERNEL_STARTUP_RETRIES', '3'))  # Default 3 retries

def cleanup_expired_kernels():
    """Remove kernels that haven't been used recently"""
    current_time = time.time()
    with kernel_lock:
        expired_kernels = []
        for session_id, kernel_data in kernels.items():
            if current_time - kernel_data['last_used'] > KERNEL_TIMEOUT:
                expired_kernels.append(session_id)

        for session_id in expired_kernels:
            try:
                kernels[session_id]['kernel_manager'].shutdown_kernel()
                del kernels[session_id]
                print(f"Cleaned up expired kernel: {session_id}")
            except Exception as e:
                print(f"Error cleaning up kernel {session_id}: {e}")

def get_or_create_kernel(session_id):
    """Get existing kernel or create a new one with retry logic"""
    current_time = time.time()
    with kernel_lock:
        if session_id not in kernels:
            # Add small random delay to reduce thundering herd when many sessions start simultaneously
            import random
            startup_delay = random.uniform(0.1, 0.5)
            time.sleep(startup_delay)

            # Try to create kernel with retries
            for attempt in range(KERNEL_STARTUP_RETRIES):
                try:
                    print(f"Creating kernel for session {session_id} (attempt {attempt + 1}/{KERNEL_STARTUP_RETRIES})")

                    # Add delay between retries to avoid thundering herd
                    if attempt > 0:
                        retry_delay = 1 + random.uniform(0, 1)  # 1-2 seconds
                        time.sleep(retry_delay)

                    # Create a new kernel manager and start the kernel
                    kernel_manager = jupyter_client.KernelManager(kernel_name='python3')
                    kernel_manager.start_kernel()

                    # Create a client to communicate with the kernel
                    kernel_client = kernel_manager.client()
                    kernel_client.start_channels()

                    # Wait for kernel to be ready with configurable timeout
                    try:
                        kernel_client.wait_for_ready(timeout=KERNEL_READY_TIMEOUT)
                    except RuntimeError:
                        print(f"Kernel {session_id} failed to start (attempt {attempt + 1})")
                        try:
                            kernel_manager.shutdown_kernel()
                        except:
                            pass
                        if attempt == KERNEL_STARTUP_RETRIES - 1:  # Last attempt
                            raise
                        continue  # Try again

                    kernels[session_id] = {
                        'kernel_manager': kernel_manager,
                        'kernel_client': kernel_client,
                        'created': current_time,
                        'last_used': current_time
                    }
                    print(f"Created new kernel for session: {session_id}")
                    break  # Success

                except Exception as e:
                    print(f"Failed to create kernel for session {session_id} (attempt {attempt + 1}): {e}")
                    if attempt == KERNEL_STARTUP_RETRIES - 1:  # Last attempt
                        # Basic error info on final failure
                        try:
                            from jupyter_client import kernelspec
                            specs = kernelspec.find_kernel_specs()
                            print(f"Available kernel specs: {list(specs.keys())}")
                        except Exception as list_error:
                            print(f"Could not list kernel specs: {list_error}")
                        raise e
        else:
            kernels[session_id]['last_used'] = current_time

        return kernels[session_id]

def execute_python_session_jupyter(generated_code, session_id, timeout=30):
    """Execute Python code in a persistent Jupyter kernel session"""
    try:
        # Clean up expired kernels periodically
        cleanup_expired_kernels()

        # Get or create kernel for this session
        kernel_data = get_or_create_kernel(session_id)
        kernel_client = kernel_data['kernel_client']

        # Execute the code
        msg_id = kernel_client.execute(generated_code)

        # Collect the results
        stdout_content = []
        stderr_content = []

        # Wait for execution to complete and collect outputs
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}

            try:
                # Get messages from kernel
                msg = kernel_client.get_iopub_msg(timeout=1)

                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg['header']['msg_type']
                    content = msg['content']

                    if msg_type == 'stream':
                        if content['name'] == 'stdout':
                            stdout_content.append(content['text'])
                        elif content['name'] == 'stderr':
                            stderr_content.append(content['text'])
                    elif msg_type == 'error':
                        stderr_content.append(f"{content['ename']}: {content['evalue']}\n")
                        stderr_content.extend(content.get('traceback', []))
                    elif msg_type == 'execute_result':
                        if 'data' in content and 'text/plain' in content['data']:
                            stdout_content.append(content['data']['text/plain'])
                    elif msg_type == 'display_data':
                        if 'data' in content and 'text/plain' in content['data']:
                            stdout_content.append(content['data']['text/plain'])
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        # Execution completed
                        break

            except queue.Empty:
                # Check if kernel is still alive
                if not kernel_client.is_alive():
                    return {"process_status": "error", "stdout": "", "stderr": "Kernel died during execution\n"}
                continue
            except Exception as e:
                return {"process_status": "error", "stdout": "", "stderr": f"Error communicating with kernel: {e}\n"}

        # Determine process status
        process_status = "completed" if not stderr_content else "error"
        if any("Error" in err or "Exception" in err for err in stderr_content):
            process_status = "error"

        return {
            "process_status": process_status,
            "stdout": "".join(stdout_content),
            "stderr": "".join(stderr_content)
        }

    except Exception as e:
        return {"process_status": "error", "stdout": "", "stderr": f"Session error: {e}\n"}

# ============================================================================
# EXEC BACKEND (Fast startup, limited features but very reliable)
# ============================================================================

# Global dictionary to store session globals by session_id
exec_sessions = {}
exec_session_lock = threading.Lock()
EXEC_SESSION_TIMEOUT = 3600  # 1 hour timeout

def cleanup_expired_exec_sessions():
    """Remove exec sessions that haven't been used recently"""
    current_time = time.time()
    with exec_session_lock:
        expired_sessions = []
        for session_id, session_data in exec_sessions.items():
            if current_time - session_data['last_used'] > EXEC_SESSION_TIMEOUT:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            try:
                del exec_sessions[session_id]
                print(f"Cleaned up expired exec session: {session_id}")
            except Exception as e:
                print(f"Error cleaning up exec session {session_id}: {e}")

def get_or_create_exec_session(session_id):
    """Get existing exec session or create a new one (instant)"""
    current_time = time.time()
    with exec_session_lock:
        if session_id not in exec_sessions:
            # Create new session with basic globals
            exec_sessions[session_id] = {
                'globals': {'__builtins__': __builtins__},
                'created': current_time,
                'last_used': current_time
            }
            print(f"Created new exec session: {session_id}")
        else:
            exec_sessions[session_id]['last_used'] = current_time

        return exec_sessions[session_id]

def execute_python_session_exec(generated_code, session_id, timeout=30):
    """Execute Python code in a persistent exec session (fast startup)"""
    try:
        # Clean up expired sessions periodically
        cleanup_expired_exec_sessions()

        # Get or create session
        session_data = get_or_create_exec_session(session_id)
        session_globals = session_data['globals']

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # Execute code with session persistence
            exec(generated_code, session_globals)

            stdout_result = stdout_capture.getvalue()
            stderr_result = stderr_capture.getvalue()

            # Determine status
            process_status = "completed" if not stderr_result else "error"

            return {
                "process_status": process_status,
                "stdout": stdout_result,
                "stderr": stderr_result
            }

        except Exception as e:
            return {
                "process_status": "error",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + f"\n{type(e).__name__}: {e}"
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        return {"process_status": "error", "stdout": "", "stderr": f"Session error: {e}\n"}

# ============================================================================
# IPYTHON BACKEND (Middle ground - faster than Jupyter, more features than exec)
# ============================================================================

try:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython not available, falling back to other backends")

# Global dictionary to store IPython shells by session_id
ipython_sessions = {}
ipython_session_lock = threading.Lock()
IPYTHON_SESSION_TIMEOUT = 3600  # 1 hour timeout

def cleanup_expired_ipython_sessions():
    """Remove IPython sessions that haven't been used recently"""
    current_time = time.time()
    with ipython_session_lock:
        expired_sessions = []
        for session_id, session_data in ipython_sessions.items():
            if current_time - session_data['last_used'] > IPYTHON_SESSION_TIMEOUT:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            try:
                del ipython_sessions[session_id]
                print(f"Cleaned up expired IPython session: {session_id}")
            except Exception as e:
                print(f"Error cleaning up IPython session {session_id}: {e}")

def get_or_create_ipython_session(session_id):
    """Get existing IPython session or create a new one (fast startup)"""
    current_time = time.time()
    with ipython_session_lock:
        if session_id not in ipython_sessions:
            try:
                # Create NEW IPython shell instance for each session (not singleton!)
                shell = TerminalInteractiveShell()  # Remove .instance() to create separate instances
                shell.init_create_namespaces()     # Initialize the shell properly

                ipython_sessions[session_id] = {
                    'shell': shell,
                    'created': current_time,
                    'last_used': current_time
                }
                print(f"Created new IPython session: {session_id}")
            except Exception as e:
                print(f"Failed to create IPython session {session_id}: {e}")
                raise
        else:
            ipython_sessions[session_id]['last_used'] = current_time

        return ipython_sessions[session_id]

def execute_python_session_ipython(generated_code, session_id, timeout=30):
    """Execute Python code in a persistent IPython session (faster than Jupyter)"""
    if not IPYTHON_AVAILABLE:
        return {"process_status": "error", "stdout": "", "stderr": "IPython backend not available\n"}

    try:
        # Clean up expired sessions periodically
        cleanup_expired_ipython_sessions()

        # Get or create session
        session_data = get_or_create_ipython_session(session_id)
        shell = session_data['shell']

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # Execute code with IPython's run_cell (handles magics, etc.)
            result = shell.run_cell(generated_code)

            stdout_result = stdout_capture.getvalue()
            stderr_result = stderr_capture.getvalue()

            # Add execution result if present
            if result.result is not None:
                stdout_result += str(result.result) + '\n'

            # Check for execution errors
            if result.error_before_exec or result.error_in_exec:
                process_status = "error"
                if result.error_in_exec:
                    stderr_result += f"\n{result.error_in_exec}"
            else:
                process_status = "completed" if not stderr_result else "error"

            return {
                "process_status": process_status,
                "stdout": stdout_result,
                "stderr": stderr_result
            }

        except Exception as e:
            return {
                "process_status": "error",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + f"\n{type(e).__name__}: {e}"
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        return {"process_status": "error", "stdout": "", "stderr": f"Session error: {e}\n"}

# ============================================================================
# UNIFIED SESSION EXECUTION (Backend dispatcher)
# ============================================================================

def execute_python_session(generated_code, session_id, timeout=30):
    """Execute Python code using the configured session backend"""
    if SESSION_BACKEND == 'jupyter':
        return execute_python_session_jupyter(generated_code, session_id, timeout)
    elif SESSION_BACKEND == 'exec':
        return execute_python_session_exec(generated_code, session_id, timeout)
    elif SESSION_BACKEND == 'ipython':
        return execute_python_session_ipython(generated_code, session_id, timeout)
    else:
        return {"process_status": "error", "stdout": "", "stderr": f"Unknown session backend: {SESSION_BACKEND}\n"}

# ============================================================================
# ORIGINAL NON-SESSION EXECUTION
# ============================================================================

def execute_python(generated_code, timeout):
    # running in a separate process to ensure any kind of crashes are properly handled
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():  # didn't finish successfully
        process.kill()
        return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}

    return queue.get()


def execute_lean4(generated_code, timeout):
    temp_file_name = None
    try:
        project_path = "/lean4/my_project"
        with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(generated_code.encode('utf-8'))

        result = subprocess.run(
            ['lake', 'env', '--dir', project_path, 'lean', temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=project_path,  # Ensure we are in the correct working directory
        )

        if result.returncode == 0:
            process_status = "completed"
        else:
            process_status = "failed"

        return {
            "process_status": process_status,
            "stdout": result.stdout.decode('utf-8'),
            "stderr": result.stderr.decode('utf-8'),
        }

    except subprocess.TimeoutExpired:
        return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"process_status": "error", "stdout": "", "stderr": str(e) + "\n"}
    finally:
        # Safely remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


# need to memory-limit to avoid common errors of allocating too much
# but this has to be done in a subprocess to not crush server itself
def execute_code_subprocess(generated_code, queue):
    limit = 1024 * 1024 * 1024 * 10  # 10gb - somehow with a smaller limit the server dies when numpy is used
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    # this can be overriden inside generated code, so it's not a guaranteed protection
    sys.stdout = StringIO()
    try:
        exec(generated_code, {})
        queue.put({"process_status": "completed", "stdout": sys.stdout.getvalue(), "stderr": ""})
    except Exception as e:
        print(f"Error: {str(e)}")
        queue.put({"process_status": "error", "stdout": "", "stderr": str(e) + "\n"})


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    generated_code = request.json['generated_code']
    timeout = request.json['timeout']
    language = request.json.get('language', 'python')

    # Get session_id from JSON body first, then from header (for nginx compatibility)
    session_id = request.json.get('session_id') or request.headers.get('X-Session-ID')

    if language == 'python':
        if session_id:
            return execute_python_session(generated_code, session_id, timeout)
        else:
            return execute_python(generated_code, timeout)
    elif language == 'lean4':
        return execute_lean4(generated_code, timeout)


# Session management endpoints
@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions"""
    session_info = {}

    if SESSION_BACKEND == 'jupyter':
        with kernel_lock:
            for session_id, kernel_data in kernels.items():
                session_info[session_id] = {
                    'backend': 'jupyter',
                    'created': kernel_data['created'],
                    'last_used': kernel_data['last_used'],
                    'alive': kernel_data['kernel_client'].is_alive()
                }
    elif SESSION_BACKEND == 'exec':
        with exec_session_lock:
            for session_id, session_data in exec_sessions.items():
                session_info[session_id] = {
                    'backend': 'exec',
                    'created': session_data['created'],
                    'last_used': session_data['last_used'],
                    'alive': True  # exec sessions are always "alive"
                }
    elif SESSION_BACKEND == 'ipython':
        with ipython_session_lock:
            for session_id, session_data in ipython_sessions.items():
                session_info[session_id] = {
                    'backend': 'ipython',
                    'created': session_data['created'],
                    'last_used': session_data['last_used'],
                    'alive': True  # IPython sessions are always "alive"
                }

    return {"sessions": session_info, "backend": SESSION_BACKEND}


@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a specific session"""
    if SESSION_BACKEND == 'jupyter':
        with kernel_lock:
            if session_id in kernels:
                try:
                    kernels[session_id]['kernel_manager'].shutdown_kernel()
                    del kernels[session_id]
                    return {"message": f"Jupyter session {session_id} deleted successfully"}
                except Exception as e:
                    return {"error": f"Error deleting Jupyter session {session_id}: {e}"}, 500
            else:
                return {"error": f"Jupyter session {session_id} not found"}, 404
    elif SESSION_BACKEND == 'exec':
        with exec_session_lock:
            if session_id in exec_sessions:
                try:
                    del exec_sessions[session_id]
                    return {"message": f"Exec session {session_id} deleted successfully"}
                except Exception as e:
                    return {"error": f"Error deleting exec session {session_id}: {e}"}, 500
            else:
                return {"error": f"Exec session {session_id} not found"}, 404
    elif SESSION_BACKEND == 'ipython':
        with ipython_session_lock:
            if session_id in ipython_sessions:
                try:
                    del ipython_sessions[session_id]
                    return {"message": f"IPython session {session_id} deleted successfully"}
                except Exception as e:
                    return {"error": f"Error deleting IPython session {session_id}: {e}"}, 500
            else:
                return {"error": f"IPython session {session_id} not found"}, 404
    else:
        return {"error": f"Unknown backend: {SESSION_BACKEND}"}, 500


# ============================================================================
# AUTOMATIC BACKGROUND CLEANUP
# ============================================================================

def background_cleanup():
    """Background thread that cleans up expired sessions every 5 minutes"""
    import time
    while True:
        try:
            time.sleep(300)  # 5 minutes
            print("Running background session cleanup...")

            if SESSION_BACKEND == 'jupyter':
                cleanup_expired_kernels()
            elif SESSION_BACKEND == 'exec':
                cleanup_expired_exec_sessions()
            elif SESSION_BACKEND == 'ipython':
                cleanup_expired_ipython_sessions()

        except Exception as e:
            print(f"Error in background cleanup: {e}")

# Start background cleanup thread
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()
print(f"Started background cleanup thread for {SESSION_BACKEND} backend")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for load balancer"""
    return {"status": "healthy", "port": os.environ.get('FLASK_PORT', '6000')}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Local sandbox server')
    parser.add_argument('--port', type=int, default=6000, help='Port to run server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # Also check environment variable (used by load balancer)
    port = int(os.environ.get('FLASK_PORT', args.port))

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    print(f"Starting sandbox server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=args.debug)
