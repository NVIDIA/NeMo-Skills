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

# Global dictionary to store Jupyter kernels by session_id
kernels = {}
kernel_lock = threading.Lock()
KERNEL_TIMEOUT = 3600  # 1 hour timeout for kernels

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
    """Get existing kernel or create a new one"""
    current_time = time.time()
    with kernel_lock:
        if session_id not in kernels:
            # Create a new kernel manager and start the kernel
            kernel_manager = jupyter_client.KernelManager()
            kernel_manager.start_kernel()

            # Create a client to communicate with the kernel
            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            # Wait for kernel to be ready
            try:
                kernel_client.wait_for_ready(timeout=30)
            except RuntimeError:
                print(f"Kernel {session_id} failed to start")
                kernel_manager.shutdown_kernel()
                raise

            kernels[session_id] = {
                'kernel_manager': kernel_manager,
                'kernel_client': kernel_client,
                'created': current_time,
                'last_used': current_time
            }
            print(f"Created new kernel for session: {session_id}")
        else:
            kernels[session_id]['last_used'] = current_time

        return kernels[session_id]

def execute_python_session(generated_code, session_id, timeout=30):
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
    session_id = request.json.get('session_id')

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
    with kernel_lock:
        session_info = {}
        for session_id, kernel_data in kernels.items():
            session_info[session_id] = {
                'created': kernel_data['created'],
                'last_used': kernel_data['last_used'],
                'alive': kernel_data['kernel_client'].is_alive()
            }
        return {"sessions": session_info}


@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a specific session"""
    with kernel_lock:
        if session_id in kernels:
            try:
                kernels[session_id]['kernel_manager'].shutdown_kernel()
                del kernels[session_id]
                return {"message": f"Session {session_id} deleted successfully"}
            except Exception as e:
                return {"error": f"Error deleting session {session_id}: {e}"}, 500
        else:
            return {"error": f"Session {session_id} not found"}, 404


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app.run(port=6000)
