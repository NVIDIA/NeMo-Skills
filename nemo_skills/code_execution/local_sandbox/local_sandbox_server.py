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
import signal
from io import StringIO
import psutil

from flask import Flask, request

app = Flask(__name__)

MEM_LIMIT_BYTES = int(os.environ.get('NEMO_SKILLS_SANDBOX_MEM_LIMIT', 10 * 1024 ** 3))  # 10 GiB default

# Code to kill the process tree for lean4 code execution
def kill_process_tree(proc):
    """
    Safely and aggressively kills a process and all its descendants.
    This is the recommended approach for ensuring cleanup.
    """
    try:
        parent = psutil.Process(proc.pid)
        # Get all children of the process, recursively.
        children = parent.children(recursive=True)
        # Add the parent to the list of processes to be killed.
        all_processes = children + [parent]
        
        # Kill all processes in the tree.
        for p in all_processes:
            try:
                # SIGKILL is a forceful, non-ignorable kill signal.
                p.kill()
            except psutil.NoSuchProcess:
                # The process might have already died, which is fine.
                pass
        
        # Wait for all processes to be terminated.
        gone, alive = psutil.wait_procs(all_processes, timeout=3)
        if alive:
            # If any processes are still alive, they are likely zombies
            # or in an unkillable state. This is a last resort.
            for p in alive:
                print(f"Warning: Process {p.pid} could not be killed.")
    except psutil.NoSuchProcess:
        # The main process already died before we could kill it.
        pass
    except Exception as e:
        print(f"Error in kill_process_tree: {e}")

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
    proc = None # <-- Keep track of the process object
    try:
        project_path = "/lean4/my_project"
        # Use a with statement for the temp file to ensure it's closed
        with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(generated_code.encode('utf-8'))
            temp_file.flush() # Ensure data is written to disk

        # Use subprocess.Popen for more control
        proc = subprocess.Popen(
            ['lake', 'env', '--dir', project_path, 'lean', temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_path,
            preexec_fn=os.setsid
        )

        # Communicate with the process, which waits for it to finish
        # This will raise TimeoutExpired if the timeout is reached
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode == 0:
            process_status = "completed"
        else:
            process_status = "failed"

        return {
            "process_status": process_status,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8'),
        }

    except subprocess.TimeoutExpired:

        # kill the process tree
        kill_process_tree(proc)
        
        # Now we can safely get any output that was generated before the kill.
        stdout, stderr = proc.communicate()

        final_stderr = stderr.decode('utf-8') + "Timed out\n"
        return {
            "process_status": "timeout",
            "stdout": stdout.decode('utf-8'),
            "stderr": final_stderr,
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"process_status": "error", "stdout": "", "stderr": str(e) + "\n"}
    finally:
        # Safely remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


def execute_shell(command, timeout):
    tmp_path = None
    try:
        # Write the full script to a temp file so /bin/bash can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sh", mode="w") as tmp:
            tmp.write(command)
            tmp_path = tmp.name
        os.chmod(tmp_path, 0o755)

        result = subprocess.run(
            ["/bin/bash", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            preexec_fn=set_limits,
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timed out\n"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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

    if language == 'ipython':
        return execute_ipython(generated_code, timeout)
    elif language == 'lean4':
        return execute_lean4(generated_code, timeout)
    elif language == 'shell':
        return execute_shell(generated_code, timeout)
    else:
        return execute_python(generated_code, std_input, timeout, language)


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app.run(port=6000)