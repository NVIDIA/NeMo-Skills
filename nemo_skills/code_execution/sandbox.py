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

import abc
import asyncio
import glob
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
import tqdm

from nemo_skills.code_execution.utils import clean_formal_generation
from nemo_skills.dataset.utils import get_lean4_header
from nemo_skills.utils import get_logger_name, python_doc_to_cmd_help, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def unroll_files(input_files):
    for manifest_pattern in input_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def extract_proof_only(lean_code: str) -> str:
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r'^\s*(theorem|example)\b')
    header_start_idx = None

    # 1. Find where the theorem starts
    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    # 2. Find where ':=' occurs, starting from the header
    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    # 3. Extract the line after ':='
    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    # 4. Collect proof lines
    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    # 5. Remove leading 'by' (with or without indentation)
    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]  # Strip 'by '

    return "\n".join(proof_lines).rstrip()


class Sandbox(abc.ABC):
    """Code execution sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        # Create async HTTP client with high limits
        self.http_session = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=2048, max_connections=2048),
        )
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        # will keep state of code sessions
        self.sessions = {}

    def clear_session(self, session_id):
        del self.sessions[session_id]

    async def close(self):
        """Close the HTTP session."""
        await self.http_session.aclose()

    async def _send_request(self, request, timeout):
        if self.ssh_server and self.ssh_key_path:
            # For SSH tunneling, use threads since there's no async version
            import sshtunnel_requests

            def ssh_request():
                sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
                return sshtunnel_request.post(
                    url=self._get_execute_url(),
                    data=json.dumps(request),
                    timeout=timeout,
                    headers={"Content-Type": "application/json"},
                )

            # Native async requires more lines of code, so we use to_thread
            # Should be ok since this is a debug mode
            output = await asyncio.to_thread(ssh_request)
        else:
            output = await self.http_session.post(
                url=self._get_execute_url(),
                content=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        # retrying 502 errors
        if output.status_code == 502:
            raise httpx.TimeoutException("502 error")
        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output):
        pass

    @abc.abstractmethod
    def _get_execute_url(self):
        pass

    @abc.abstractmethod
    def _prepare_request(self, generated_code, timeout):
        pass

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = 'ipython',
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
        traceback_verbosity='plain',  # could be plain, context, verbose, or minimal
    ) -> Tuple[Dict, str]:
        traceback_verbosity = traceback_verbosity.capitalize()
        if session_id is None and language == "ipython":  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []

        if session_id is not None:
            self.sessions[session_id].append(generated_code)

        if language == 'ipython':
            TO_EXECUTE = """
import traceback
import json
import os
import re
import warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '16'

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io

def simplify_errors(error_text):
    def strip_ansi_codes(text):
        ansi_escape = re.compile(r'\\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    error_text = strip_ansi_codes(error_text)
    output = []
    for line in error_text.split('\\n'):
        if line.strip().startswith('File <ipython-'):
            continue
        output.append(line)
    return '\\n'.join(output)

code_snippets = []
"""
            for code_snippet in self.sessions[session_id]:
                TO_EXECUTE += f'\ncode_snippets.append({repr(code_snippet)})\n'

            # we do `strip() + \\n` below to ensure that `print(res)` and `res` return the same output
            TO_EXECUTE += f"""
try:
    shell = InteractiveShell()
    shell.InteractiveTB.set_mode(mode='{traceback_verbosity}')
    for code in code_snippets:
        with io.capture_output() as captured:
            exec_result = shell.run_cell(code)
    stdout = captured.stdout.replace("Out[1]: ", "").strip()
    stderr = captured.stderr.replace("Out[1]: ", "").strip()
    if len(stdout) > {max_output_characters}:
        stdout = stdout[:{max_output_characters}] + "<output cut>"
    if len(stderr) > {max_output_characters}:
        stderr = stderr[:{max_output_characters}] + "<output cut>"
    if stdout:
        if '{traceback_verbosity}' in ['Minimal', 'Plain']:
            stdout = simplify_errors(stdout)
        stdout += "\\n"
    if stderr:
        if '{traceback_verbosity}' in ['Minimal', 'Plain']:
            stderr = simplify_errors(stderr)
        stderr += "\\n"
    has_error = exec_result.error_before_exec or exec_result.error_in_exec
    to_return = {{"process_status": "error" if has_error else "completed", "stdout": stdout, "stderr": stderr}}

except Exception:
    # removing useless prefix from traceback
    to_return = {{
        "process_status": "error",
        "stdout": "",
        "stderr": "\\n".join(traceback.format_exc().split("\\n")[3:]),
    }}
print(json.dumps(to_return))
"""
        elif language in ["python", "pypy3", "python3", "lean4", "shell"]:
            if session_id is not None:
                raise RuntimeError(
                    f"Stateful execution for {language} is not supported. session_id is {session_id} but should be None"
                )
            TO_EXECUTE = generated_code
        else:
            raise ValueError(f"Unsupported language: {language}")

        request = self._prepare_request(TO_EXECUTE, timeout, language, std_input)
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            output = {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
        # removing last state to not re-execute code with errors
        if session_id is not None:
            if output['process_status'] != "completed":
                self.sessions[session_id] = self.sessions[session_id][:-1]
        return output, session_id

    async def is_proof_correct(self, pred_output, timeout=30.0):
        TO_EXECUTE = pred_output

        request = self._prepare_request(TO_EXECUTE, timeout, "lean4")
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            return "timeout"
        if output['process_status'] == 'completed' and output['stdout'] != '':
            return 'has_sorry'
        return output["process_status"]

    async def batch_evaluate_results(
        self,
        input_files: List[str],
        num_parallel_requests=10,
        timeout=30.0,
        answer_format="lean4-proof",
        use_predicted_proof_key: bool = False,
        final_answer_key: str = "**FINAL ANSWER**",
        restate_formal_statement: bool = True,
        strip_theorem_from_proof: bool = True,
    ):
        """Evaluate results and write back to original files."""

        semaphore = asyncio.Semaphore(num_parallel_requests)

        async def process_line(line_data):
            """Process a single line and return updated line data."""
            if not line_data or not line_data.strip():
                return line_data

            line_dict = json.loads(line_data)
            if not line_dict:
                return line_data

            # Prepare predicted_proof based on format
            if answer_format == "lean4-proof":
                if not use_predicted_proof_key:
                    generation = clean_formal_generation(line_dict["generation"], final_answer_key=final_answer_key)
                    line_dict["predicted_proof"] = (
                        line_dict["header"]
                        + (line_dict["formal_statement"] if restate_formal_statement else '')
                        + extract_proof_only(generation)
                        if strip_theorem_from_proof
                        else generation
                    )
                else:
                    if "predicted_proof" not in line_dict:
                        raise ValueError(
                            "predicted_proof key not found in the line_dict. "
                            "Set use_predicted_proof_key=False to re-combine"
                        )
            elif answer_format == "lean4-statement":
                if not use_predicted_proof_key:
                    generation = clean_formal_generation(line_dict["generation"])
                    header = get_lean4_header()
                    line_dict["predicted_proof"] = header + generation + "\n sorry"
                else:
                    if "predicted_proof" not in line_dict:
                        raise ValueError(
                            "predicted_proof key not found in the line_dict. "
                            "Set use_predicted_proof_key=False to re-combine"
                        )
            else:
                raise ValueError(f'Unknown answer_format: {answer_format}')

            # Evaluate proof with concurrency control
            async with semaphore:
                proof_status = await self.is_proof_correct(line_dict["predicted_proof"], timeout=timeout)
                line_dict["proof_status"] = proof_status

            return json.dumps(line_dict)

        # Process each file
        for input_file in unroll_files(input_files):
            # Read all lines
            with open(input_file, "rt", encoding="utf-8") as f:
                lines = f.readlines()

            # Process lines concurrently with progress bar
            print(f"Processing {input_file}...")
            processed_lines = []
            for line in tqdm.tqdm(lines):
                result = await process_line(line.rstrip('\n'))
                processed_lines.append(result)

            # Write to temp file then replace original
            temp_file = input_file + "-tmp"
            with open(temp_file, "wt", encoding="utf-8") as f:
                for line in processed_lines:
                    f.write(line + "\n")

            # Replace original with temp file
            os.replace(temp_file, input_file)


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self):
        return f"http://{self.host}:{self.port}/execute"

    def _parse_request_output(self, output):
        try:
            return output.json()
        except json.JSONDecodeError:
            LOG.error("Error during parsing output: %s", output.text)
            return {'process_status': 'error', 'stdout': '', 'stderr': 'Unknown error'}

    def _prepare_request(self, generated_code, timeout, language='ipython', std_input=""):
        return {
            "generated_code": generated_code,
            "std_input": std_input,
            "timeout": timeout,
            "language": language,
        }


sandboxes = {
    'local': LocalSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f'\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}'
    return python_doc_to_cmd_help(Sandbox, docs_prefix=prefix, arg_prefix="sandbox.")
