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
import signal
from io import StringIO
from IPython.terminal.interactiveshell import TerminalInteractiveShell

from flask import Flask, request

app = Flask(__name__)

# Global dictionary to store IPython shells by session_id
sessions = {}
session_lock = threading.Lock()
SESSION_TIMEOUT = float(os.getenv('SANDBOX_SESSION_TIMEOUT', -1))

def cleanup_expired_sessions():
    """Remove IPython sessions that haven't been used recently"""
    current_time = time.time()
    with session_lock:
        expired_sessions = []
        for session_id, session_data in sessions.items():
            if current_time - session_data['last_used'] > SESSION_TIMEOUT:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            try:
                del sessions[session_id]
                print(f"Cleaned up expired session: {session_id}")
            except Exception as e:
                print(f"Error cleaning up session {session_id}: {e}")

def get_or_create_session(session_id):
    """Get existing IPython session or create a new one (fast startup)"""
    current_time = time.time()
    with session_lock:
        if session_id not in sessions:
            try:
                # Create new IPython shell instance for each session
                shell = TerminalInteractiveShell()
                shell.init_create_namespaces()     # Initialize the shell properly

                sessions[session_id] = {
                    'shell': shell,
                    'created': current_time,
                    'last_used': current_time
                }
                print(f"Created new IPython session: {session_id}")
            except Exception as e:
                print(f"Failed to create IPython session {session_id}: {e}")
                raise
        else:
            sessions[session_id]['last_used'] = current_time

        return sessions[session_id]

def execute_ipython_session(generated_code, session_id, timeout=30):
    """Execute Python code in a persistent IPython session"""
    try:
        # Clean up expired sessions periodically
        if SESSION_TIMEOUT > 0:
            cleanup_expired_sessions()

        # Get or create session
        session_data = get_or_create_session(session_id)
        shell = session_data['shell']

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # Execute code with IPython's run_cell (handles imports, variables, etc.)
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
MEM_LIMIT_BYTES = int(os.environ.get('NEMO_SKILLS_SANDBOX_MEM_LIMIT', 10 * 1024 ** 3))  # 10 GiB default

def set_limits(mem_bytes: int = MEM_LIMIT_BYTES) -> None:
    """
    Apply RLIMITs and start a new session for the child process.

    Called via `preexec_fn` (subprocess) or directly in a forked worker.
    """
    resource.setrlimit(resource.RLIMIT_AS,   (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
    os.setsid()                              # isolate PGID / signals

def execute_ipython(generated_code, timeout):
    # running in a separate process to ensure any kind of crashes are properly handled
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():  # didn't finish successfully
        process.kill()
        return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}

    return queue.get()

def execute_python(generated_code, std_input, timeout, language):

    execution_command = [language, "-c", generated_code]
    try:
        process = subprocess.Popen(
            execution_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            preexec_fn=set_limits,
        )
        stdout, stderr = process.communicate(input=std_input, timeout=timeout)
        return {"process_status": "completed", "stdout": stdout, "stderr": stderr}
    except subprocess.TimeoutExpired:
        try:
            # kill the whole process group
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)  # reap, no extra timeout needed
        return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}


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

    # this can be overriden inside generated code, so it's not a guaranteed protection
    set_limits()
    sys.stdout = StringIO()
    try:
        exec(generated_code, {})
        queue.put(sys.stdout.getvalue())
    except Exception as e:
        print(f"Error: {str(e)}")
        queue.put({"process_status": "error", "stdout": "", "stderr": str(e) + "\n"})


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    generated_code = request.json['generated_code']
    timeout = request.json['timeout']
    language = request.json.get('language', 'ipython')
    std_input = request.json.get('std_input', '')

    # Get session_id from JSON body
    session_id = request.json.get('session_id')

    if language == 'python':
        if session_id:
            return execute_python_session(generated_code, session_id, timeout)
        else:
            return execute_python(generated_code, timeout)
    elif language == 'lean4':
        return execute_lean4(generated_code, timeout)
    else:
        return execute_python(generated_code, std_input, timeout, language)


# Session management endpoints
@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active IPython sessions"""
    session_info = {}

    with session_lock:
        for session_id, session_data in sessions.items():
            session_info[session_id] = {
                'backend': 'ipython',
                'created': session_data['created'],
                'last_used': session_data['last_used'],
                'alive': True  # IPython sessions are always "alive"
            }

    return {"sessions": session_info, "backend": "ipython"}


@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a specific IPython session"""
    with session_lock:
        if session_id in sessions:
            try:
                del sessions[session_id]
                return {"message": f"IPython session {session_id} deleted successfully"}
            except Exception as e:
                return {"error": f"Error deleting IPython session {session_id}: {e}"}, 500
        else:
            return {"error": f"IPython session {session_id} not found"}, 404

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for load balancer"""
    return {"status": "healthy", "port": os.environ.get('FLASK_PORT', '6000')}


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app.run(port=6000)
