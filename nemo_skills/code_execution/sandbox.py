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
import glob
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.utils import python_doc_to_cmd_help, unroll_files
from nemo_skills.dataset.utils import get_lean4_header
from nemo_skills.code_execution.utils import clean_formal_generation

LOG = logging.getLogger(__file__)


class DummyFuture:
    def __init__(self, return_value):
        self.return_value = return_value

    def result(self):
        return self.return_value


def unroll_files(input_files):
    for manifest_pattern in input_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def cleanup_tmp_files(input_files):
    # removing any potentially present tmp files
    for manifest in unroll_files(input_files):
        try:
            os.remove(manifest + "-tmp")
        except OSError:
            pass


def dump_data(input_files, data, map_to_future, update_fn, answer_format="natural_language"):
    LOG.info("Waiting for current results and dumping to tmp files")
    tmp_file_handles = [
        open(manifest + f"-tmp", "at", encoding="utf-8", buffering=1) for manifest in unroll_files(input_files)
    ]

    for line_data in data:
        for file_data, file_handle in zip(line_data, tmp_file_handles):
            if file_data is None:
                continue
            line_dict = json.loads(file_data)
            if not line_dict:
                file_handle.write("\n")
                continue
            update_fn(map_to_future, line_dict, answer_format)
            file_handle.write(json.dumps(line_dict) + "\n")

    for file_handle in tmp_file_handles:
        file_handle.close()


def write_tmp_files_back(input_files):
    """Will gracefully handle early exits on errors by properly merging files"""
    LOG.info("Writing temporary files back into original files")
    for manifest in unroll_files(input_files):
        # copying the rest of the results unchanged if any to tmp file
        with open(manifest + "-tmp", "rt") as fin:
            processed_lines = sum(1 for _ in fin)
        with open(manifest, "rt", encoding="utf-8") as fin, open(manifest + "-tmp", "at", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin):
                if line_idx >= processed_lines:
                    fout.write(line)
        # then replacing original file with tmp file
        os.replace(manifest + "-tmp", manifest)


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
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=1500, pool_connections=1500, max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        self.http_session = session
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        # will keep state of code sessions
        self.sessions = {}

    def clear_session(self, session_id):
        del self.sessions[session_id]

    def _send_request(self, request, timeout):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            output = sshtunnel_request.post(
                url=self._get_execute_url(),
                data=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        else:
            output = self.http_session.post(
                url=self._get_execute_url(),
                data=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        # retrying 502 errors
        if output.status_code == 502:
            raise requests.exceptions.Timeout
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

    def execute_code(
        self,
        generated_code: str,
        language: str = 'python',
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict, str]:
        if session_id is None and language == "python":  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []

        if session_id is not None:
            self.sessions[session_id].append(generated_code)

        if language == 'python':
            TO_EXECUTE = """
import traceback
import json
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '16'

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io

code_snippets = []
"""
            for code_snippet in self.sessions[session_id]:
                TO_EXECUTE += f'\ncode_snippets.append({repr(code_snippet)})\n'

            # we do `strip() + \\n` below to ensure that `print(res)` and `res` return the same output
            TO_EXECUTE += f"""
try:
    shell = InteractiveShell()
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
        stdout += "\\n"
    if stderr:
        stderr += "\\n"
    to_return = {{"process_status": "completed", "stdout": stdout, "stderr": stderr}}
except Exception:
    # removing useless prefix from traceback
    to_return = {{
        "process_status": "error",
        "stdout": "",
        "stderr": "\\n".join(traceback.format_exc().split("\\n")[3:]),
    }}
print(json.dumps(to_return))
"""
        elif language == 'lean4':
            if session_id is not None:
                raise RuntimeError(
                    f"Stateful execution for {language} is not supported. session_id is {session_id} but should be None"
                )
            TO_EXECUTE = generated_code
        else:
            raise ValueError(f"Unsupported language: {language}")

        request = self._prepare_request(TO_EXECUTE, timeout, language)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
        # removing last state to not re-execute code with errors
        if session_id is not None:
            if output['stderr'] or 'Traceback (most recent call last)' in output['stdout']:
                self.sessions[session_id] = self.sessions[session_id][:-1]
        return output, session_id

    def is_output_correct(
        self,
        pred_output,
        gt_output,
        take_modulo: int | None = None,
        include_percentage=True,
        tolerance=1e-4,
        timeout=10.0,
    ):
        # embedding the full math grader code here to send to server for execution
        with open(Path(__file__).absolute().parent / "math_grader.py", "rt") as fin:
            math_grader_code = fin.read()

        # corner cases
        if isinstance(pred_output, str):
            pred_output = pred_output.replace("'''", r'\'\'\'')
            while pred_output.endswith('\\'):
                pred_output = pred_output[:-1]

        if isinstance(gt_output, str):
            gt_output = gt_output.replace("'''", r'\'\'\'')
            while gt_output.endswith('\\'):
                gt_output = gt_output[:-1]

        if take_modulo is not None:
            gt_output = int(gt_output) % take_modulo
            try:
                pred_output = int(pred_output) % take_modulo
            except:
                pred_output = None
            # no need to simpy call in this case
            return pred_output == gt_output

        math_equal_call = f"""
    output = math_equal(
        {repr(str(pred_output))},
        {repr(str(gt_output))},
        {include_percentage},
        {tolerance},
        {timeout},
    )
"""

        TO_EXECUTE = f"""
import os
import sys
import json
from io import StringIO
os.environ['OPENBLAS_NUM_THREADS'] = '16'

{math_grader_code}

stdout = sys.stdout
# removing all output to not capture that
sys.stdout = sys.stderr = StringIO()
try:
    {math_equal_call}
    error_message = ""
except Exception as e:
    output = False
    error_message = str(e)
# restoring the output to get the print
sys.stdout = stdout
print(json.dumps({{"result": output, "error_message": error_message}}))
"""

        request = self._prepare_request(TO_EXECUTE, timeout)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {'result': False, 'error_message': 'timeout'}

        if output.get('error_message', 'internal error!'):
            # logging the error
            LOG.error(
                "Error during correctness check:\noutput dict: %s\npred_output: %s\ngt_output: %s\nmath_equal_call: %s",
                output,
                pred_output,
                gt_output,
                math_equal_call,
            )

        if 'result' not in output:
            LOG.error(
                "Unexpected output:\noutput dict: %s\npred_output: %s\ngt_output: %s\nmath_equal_call: %s",
                output,
                pred_output,
                gt_output,
                math_equal_call,
            )
            return False

        return output['result']

    def is_proof_correct(self, pred_output, timeout=30.0):
        TO_EXECUTE = pred_output

        request = self._prepare_request(TO_EXECUTE, timeout, "lean4")
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            return "timeout"
        return output["process_status"]

    def batch_evaluate_results(
        self,
        input_files: List[str],
        num_parallel_requests=100,
        in_memory_lines=1500,
        include_percentage=True,
        tolerance=1e-4,
        timeout=10.0,
        take_modulo=None,
        answer_format="natural_language",
        ignore_cache: bool = False,
        use_predicted_answer_key: bool = False,
        use_predicted_proof_key: bool = False,
        extract_from_boxed: bool = True,
        extract_regex: str = r"The final answer is (.+)$",
    ):
        """Will write if the results are correct back into the original files."""

        file_handles = [open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(input_files)]
        cleanup_tmp_files(input_files)

        def update_fn(map_to_future, line_dict, answer_format="natural_language"):
            if answer_format == "natural_language":
                line_dict["is_correct"] = map_to_future[
                    (line_dict["predicted_answer"], line_dict["expected_answer"])
                ].result()
            elif "lean4-" in answer_format:
                line_dict["proof_status"] = map_to_future[(line_dict["predicted_proof"])].result()

        data = []
        with ThreadPoolExecutor(max_workers=num_parallel_requests) as executor:
            for line_idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
                if line_idx % in_memory_lines == 0:
                    if line_idx > 0:  # dumping into tmp files
                        dump_data(input_files, data, map_to_future, update_fn, answer_format)
                    # new in-memory buffer
                    data = []
                    map_to_future = {}

                data.append([])
                for file_line in lines:
                    data[-1].append(file_line)
                    if file_line is None:  # if different files have different number of lines
                        continue
                    line_dict = json.loads(file_line)
                    if not line_dict:  # can be empty for incomplete generations
                        continue
                    if answer_format == "natural_language":
                        gt_answer = line_dict["expected_answer"]

                    if answer_format == "natural_language":
                        if not use_predicted_answer_key:
                            line_dict["predicted_answer"] = extract_answer(
                                line_dict["generation"],
                                extract_from_boxed=extract_from_boxed,
                                extract_regex=extract_regex,
                            )
                        else:
                            if "predicted_answer" not in line_dict:
                                raise ValueError(
                                    "predicted_answer key not found in the line_dict. "
                                    "Set use_predicted_answer_key=False to re-extract"
                                )
                    elif answer_format == "lean4-proof":
                        if not use_predicted_proof_key:
                            generation = clean_formal_generation(line_dict["generation"])
                            line_dict["predicted_proof"] = line_dict["header"] + line_dict["formal_statement"] + generation
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
                            
                    data[-1][-1] = json.dumps(line_dict)

                    if answer_format == "natural_language":
                        predicted_answer = line_dict["predicted_answer"]
                    elif "lean4-" in answer_format:
                        predicted_proof = line_dict["predicted_proof"]

                    if answer_format == "natural_language" and (predicted_answer, gt_answer) in map_to_future:
                        continue
                    elif ("lean4-" in answer_format) and predicted_proof in map_to_future:
                        continue

                    if (
                        ignore_cache
                        or (line_dict.get("is_correct") is None and answer_format == "natural_language")
                        or (line_dict.get("proof_status") is None and ("lean4-" in answer_format))
                    ):
                        if answer_format == "natural_language":
                            map_to_future[(predicted_answer, gt_answer)] = executor.submit(
                                self.is_output_correct,
                                predicted_answer,
                                gt_answer,
                                include_percentage=include_percentage,
                                tolerance=tolerance,
                                timeout=timeout,
                                take_modulo=take_modulo,
                            )
                        elif "lean4-" in answer_format:
                            map_to_future[predicted_proof] = executor.submit(
                                self.is_proof_correct,
                                predicted_proof,
                                timeout=timeout,
                            )
                    else:
                        if answer_format == "natural_language":
                            map_to_future[(predicted_answer, gt_answer)] = DummyFuture(line_dict["is_correct"])
                        elif "lean4-" in answer_format:
                            map_to_future[predicted_proof] = DummyFuture(line_dict["proof_status"])

            for file_handle in file_handles:
                file_handle.close()

            if len(data) > 0:
                dump_data(input_files, data, map_to_future, update_fn, answer_format)

        write_tmp_files_back(input_files)


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

    def _prepare_request(self, generated_code, timeout, language='python'):
        return {
            "generated_code": generated_code,
            "timeout": timeout,
            "language": language,
        }


class PistonSandbox(Sandbox):
    """Piston sandbox (https://github.com/engineer-man/piston)"""

    def _get_execute_url(self):
        return f"{self.host}/execute"

    def _parse_request_output(self, output):
        output = output.json()
        if output['run']['signal'] == "SIGKILL":
            return {'result': None, 'error_message': 'Unknown error: SIGKILL'}
        return json.loads(output['run']['output'])

    def _prepare_request(self, generated_code, timeout):
        return {
            "language": "py",
            "version": "3.10.0",
            "files": [
                {
                    "content": generated_code,
                }
            ],
            "stdin": "",
            "args": [],
            "run_timeout": timeout * 1000.0,  # milliseconds
            "compile_memory_limit": -1,
            "run_memory_limit": -1,
        }


sandboxes = {
    'local': LocalSandbox,
    'piston': PistonSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f'\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}'
    return python_doc_to_cmd_help(Sandbox, docs_prefix=prefix, arg_prefix="sandbox.")
