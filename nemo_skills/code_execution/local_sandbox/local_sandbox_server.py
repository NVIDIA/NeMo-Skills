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
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time

import psutil
from flask import Flask, request
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils import io

app = Flask(__name__)

# Identify worker and configure logging so messages are visible per-worker
worker_id = os.getenv("WORKER_NUM", "unknown")
logging.basicConfig(
    level=logging.INFO,
    format=f"[worker {worker_id}] %(asctime)s %(levelname)s: %(message)s",
)


def log_session_count(prefix: str = "") -> None:
    try:
        logging.info("%sactive_sessions=%d", prefix, len(sessions))
    except Exception:
        pass


# Global dictionary to store IPython shells by session_id
sessions = {}
session_lock = threading.Lock()
SESSION_TIMEOUT = float(os.getenv("SANDBOX_SESSION_TIMEOUT", 1800))  # 1 hour


def cleanup_expired_sessions():
    """Remove IPython sessions that haven't been used recently"""
    current_time = time.time()
    with session_lock:
        expired_sessions = []
        for session_id, session_data in sessions.items():
            if current_time - session_data["last_used"] > SESSION_TIMEOUT:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            try:
                del sessions[session_id]
                logging.info(f"Cleaned up expired session: {session_id}")
            except Exception as e:
                logging.warning(f"Error cleaning up session {session_id}: {e}")


def get_or_create_session(session_id):
    """Get existing IPython session or create a new one (fast startup)"""
    current_time = time.time()
    new_session_created = False
    with session_lock:
        if session_id not in sessions:
            new_session_created = True
            try:
                # Create new IPython shell instance for each session
                shell = TerminalInteractiveShell()
                shell.init_create_namespaces()  # Initialize the shell properly
                shell.user_ns["_oh"] = {}

                sessions[session_id] = {"shell": shell, "created": current_time, "last_used": current_time}
                logging.info("Created new IPython session: %s (total=%d)", session_id, len(sessions))
                log_session_count()
            except Exception as e:
                logging.error("Failed to create IPython session %s: %s", session_id, e)
                raise
        else:
            sessions[session_id]["last_used"] = current_time

        return sessions[session_id], new_session_created


def postprocess_output(output, traceback_verbosity):
    if traceback_verbosity not in ["Minimal", "Plain"]:
        return output

    def strip_ansi_codes(text):
        ansi_escape = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    output = strip_ansi_codes(output)
    lines = []
    for line in output.split("\n"):
        if line.strip().startswith("File <ipython-"):
            continue
        line = re.sub(r"^Out\[\d+\]:\s*", "", line)
        lines.append(line)

    # Remove leading blank lines introduced by displayhook
    while lines and lines[0] == "":
        lines.pop(0)

    output = "\n".join(lines).rstrip()
    return output + ("\n" if output else "")


def execute_ipython_session(generated_code, session_id, traceback_verbosity="Plain"):
    """Execute Python code in a persistent IPython session"""
    try:
        # Clean up expired sessions periodically
        if SESSION_TIMEOUT > 0:
            cleanup_expired_sessions()

        # Get or create session
        session_data, new_session_created = get_or_create_session(session_id)
        shell = session_data["shell"]
        shell.InteractiveTB.set_mode(mode=traceback_verbosity)

        # Capture stdout/stderr
        try:
            with io.capture_output() as captured:
                result = shell.run_cell(generated_code)
            stdout_result = captured.stdout
            stderr_result = captured.stderr
            has_error = result.error_before_exec or result.error_in_exec
            process_status = "completed" if not has_error else "error"

        except Exception as e:
            process_status = "error"
            stdout_result = captured.stdout if "captured" in locals() else ""
            stderr_extra = f"\n{type(e).__name__}: {e}"
            stderr_result = (captured.stderr if "captured" in locals() else "") + stderr_extra
            stdout_result += stderr_result
            stderr_result = ""

        return {
            "process_status": process_status,
            "stdout": postprocess_output(stdout_result, traceback_verbosity),
            "stderr": postprocess_output(stderr_result, traceback_verbosity),
            "new_session_created": new_session_created,
        }

    except Exception as e:
        return {
            "process_status": "error",
            "stdout": "",
            "stderr": f"Session error: {e}\n",
            "new_session_created": new_session_created,
        }


# Log per-request session count after each response
@app.after_request
def _after_log_session_count(response):
    log_session_count()
    return response


MEM_LIMIT_BYTES = int(os.environ.get("NEMO_SKILLS_SANDBOX_MEM_LIMIT", 10 * 1024**3))  # 10 GiB default

# Set per-worker memory limit for ipython session
resource.setrlimit(resource.RLIMIT_AS, (2 * MEM_LIMIT_BYTES, 2 * MEM_LIMIT_BYTES))
resource.setrlimit(resource.RLIMIT_DATA, (2 * MEM_LIMIT_BYTES, 2 * MEM_LIMIT_BYTES))
logging.info("Applied worker memory limit (RLIMIT_AS/RLIMIT_DATA): %d bytes", 2 * MEM_LIMIT_BYTES)


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
                logging.warning("Process %s could not be killed.", p.pid)
    except psutil.NoSuchProcess:
        # The main process already died before we could kill it.
        pass
    except Exception as e:
        logging.error("Error in kill_process_tree: %s", e)


def set_limits(mem_bytes: int = MEM_LIMIT_BYTES) -> None:
    """
    Apply RLIMITs and start a new session for the child process.

    Called via `preexec_fn` (subprocess) or directly in a forked worker.
    """
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
    os.setsid()  # isolate PGID / signals


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
    proc = None  # <-- Keep track of the process object
    try:
        project_path = "/lean4/my_project"
        # Use a with statement for the temp file to ensure it's closed
        with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(generated_code.encode("utf-8"))
            temp_file.flush()  # Ensure data is written to disk

        # Use subprocess.Popen for more control
        proc = subprocess.Popen(
            ["lake", "env", "--dir", project_path, "lean", temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_path,
            preexec_fn=os.setsid,
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
            "stdout": stdout.decode("utf-8"),
            "stderr": stderr.decode("utf-8"),
        }

    except subprocess.TimeoutExpired:
        # kill the process tree
        kill_process_tree(proc)
        # Now we can safely get any output that was generated before the kill.
        stdout, stderr = proc.communicate()

        final_stderr = stderr.decode("utf-8") + "Timed out\n"
        return {
            "process_status": "timeout",
            "stdout": stdout.decode("utf-8"),
            "stderr": final_stderr,
        }

    except Exception as e:
        logging.error("Error executing Lean4 code: %s", e)
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


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    generated_code = request.json["generated_code"]
    timeout = request.json["timeout"]
    language = request.json.get("language", "ipython")
    std_input = request.json.get("std_input", "")
    max_output_characters = request.json.get("max_output_characters", 1000)
    traceback_verbosity = request.json.get("traceback_verbosity", "Plain")

    session_id = request.headers.get("X-Session-ID")

    if language == "ipython":
        if session_id is None:
            return {"error": "X-Session-ID header required for ipython sessions"}, 400
        result = execute_ipython_session(generated_code, session_id, traceback_verbosity)
    elif language == "lean4":
        result = execute_lean4(generated_code, timeout)
    elif language == "shell":
        return execute_shell(generated_code, timeout)
    else:
        result = execute_python(generated_code, std_input, timeout, language)

    if len(result["stdout"]) > max_output_characters:
        result["stdout"] = result["stdout"][:max_output_characters] + "<output cut>"
    if len(result["stderr"]) > max_output_characters:
        result["stderr"] = result["stderr"][:max_output_characters] + "<output cut>"

    return result


# Session management endpoints
@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active IPython sessions"""
    try:
        session_info = {}
        with session_lock:
            # Snapshot items to avoid iteration errors if dict changes
            for session_id, session_data in list(sessions.items()):
                session_info[session_id] = {
                    "backend": "ipython",
                    "created": session_data["created"],
                    "last_used": session_data["last_used"],
                    "alive": True,  # IPython sessions are always "alive"
                }
        return {"sessions": session_info, "backend": "ipython"}
    except Exception as e:
        import traceback

        error_msg = f"Error in list_sessions: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return {"error": error_msg}, 500


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
    return {"status": "healthy", "worker": os.environ.get("WORKER_NUM", "unknown")}


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)
    app.run(port=6000)
