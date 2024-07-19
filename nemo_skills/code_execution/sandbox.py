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
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backoff
import requests

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.utils import python_doc_to_cmd_help, unroll_files

LOG = logging.getLogger(__file__)


class DummyFuture:
    def __init__(self, return_value):
        self.return_value = return_value

    def result(self):
        return self.return_value


def unroll_files(prediction_jsonl_files):
    for manifest_pattern in prediction_jsonl_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def cleanup_tmp_files(prediction_jsonl_files):
    # removing any potentially present tmp files
    for manifest in unroll_files(prediction_jsonl_files):
        try:
            os.remove(manifest + "-tmp")
        except OSError:
            pass


def dump_data(prediction_jsonl_files, data, map_to_future, update_fn):
    LOG.info("Waiting for current results and dumping to tmp files")
    tmp_file_handles = [
        open(manifest + f"-tmp", "at", encoding="utf-8", buffering=1)
        for manifest in unroll_files(prediction_jsonl_files)
    ]

    for line_data in data:
        for file_data, file_handle in zip(line_data, tmp_file_handles):
            if file_data is None:
                continue
            line_dict = json.loads(file_data)
            if not line_dict:
                file_handle.write("\n")
                continue
            update_fn(map_to_future, line_dict)
            file_handle.write(json.dumps(line_dict) + "\n")

    for file_handle in tmp_file_handles:
        file_handle.close()


def write_tmp_files_back(prediction_jsonl_files):
    """Will gracefully handle early exits on errors by properly merging files"""
    LOG.info("Writing temporary files back into original files")
    for manifest in unroll_files(prediction_jsonl_files):
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

    NOT_EXECUTED = "<not_executed>"
    EXECUTION_ERROR = "Execution error:"
    SYNTAX_ERROR = "Syntax error:"
    RESULT_NOT_DEFINED_ERROR = "Result is not defined"
    TIMEOUT_ERROR = "timeout"
    UNDEFINED_ERROR = "Undefined error:"
    ERROR_PREFIXES = (EXECUTION_ERROR, SYNTAX_ERROR, RESULT_NOT_DEFINED_ERROR, TIMEOUT_ERROR, UNDEFINED_ERROR)

    def __init__(
        self,
        host: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.http_session = requests.Session()
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        # will keep state of code sessions
        self.sessions = {}

    def clear_session(self, session_id):
        del self.sessions[session_id]

    @backoff.on_exception(backoff.constant, requests.exceptions.Timeout, interval=1, max_tries=3)
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
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict, str]:
        if session_id is None:  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []
        generated_code = generated_code.replace('"""', r'\"\"\"')
        self.sessions[session_id].append(generated_code)
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
            TO_EXECUTE += f'\ncode_snippets.append("""{code_snippet}""")\n'

        TO_EXECUTE += f"""
try:
    shell = InteractiveShell()
    for code in code_snippets:
        with io.capture_output() as captured:
            exec_result = shell.run_cell(code)
    # serializing to str to make sure things like Rational can be converted to json
    output = f"{{captured.stdout}}{{captured.stderr}}".strip().replace("Out[1]: ", "")
    if len(output) > {max_output_characters}:
        output = output[:{max_output_characters}] + "<output cut>"
    error_message = ""
    if exec_result.error_in_exec is not None:
        # full traceback will be part of output
        error_message = f"{Sandbox.EXECUTION_ERROR} {{str(exec_result.error_in_exec)}}"
    elif exec_result.error_before_exec is not None:
        # full traceback will be part of output
        error_message = f"{Sandbox.SYNTAX_ERROR} {{str(exec_result.error_before_exec)}}"
    elif output == "":
        error_message = "{Sandbox.RESULT_NOT_DEFINED_ERROR}"
    to_return = {{"result": output, "error_message": error_message}}
except Exception:
    # removing useless prefix from traceback
    to_return = {{
        "result": None,
        "error_message": "{Sandbox.UNDEFINED_ERROR}" + "\\n".join(traceback.format_exc().split("\\n")[3:]),
    }}
print(json.dumps(to_return))
"""
        request = self._prepare_request(TO_EXECUTE, timeout)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {'result': None, 'error_message': Sandbox.TIMEOUT_ERROR}
        # resetting state to not re-execute code with errors
        if output['error_message']:
            self.clear_session(session_id)
            session_id = None
        return output, session_id

    def is_output_correct(self, pred_output, gt_output, include_percentage=True, tolerance=1e-4, timeout=10.0):
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
    output = math_equal(
        r'''{pred_output}''',
        r'''{gt_output}''',
        {include_percentage},
        {tolerance},
        {timeout},
    )
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
            output = {'result': False, 'error_message': Sandbox.TIMEOUT_ERROR}
        if output['error_message']:
            # logging the error
            LOG.warning("Error during correctness check: %s", output['error_message'])

        return output['result']

    def batch_evaluate_results(
        self,
        prediction_jsonl_files: List[str],
        num_parallel_requests=100,
        in_memory_lines=1500,
        include_percentage=True,
        tolerance=1e-4,
        timeout=10.0,
        ignore_cache: bool = False,
    ):
        """Will write if the results are correct back into the original files."""
        import tqdm

        file_handles = [open(manifest, "rt", encoding="utf-8") for manifest in unroll_files(prediction_jsonl_files)]
        cleanup_tmp_files(prediction_jsonl_files)

        def update_fn(map_to_future, line_dict):
            line_dict["is_correct"] = map_to_future[
                (line_dict["predicted_answer"], line_dict["expected_answer"])
            ].result()

        data = []
        with ThreadPoolExecutor(max_workers=num_parallel_requests) as executor:
            for line_idx, lines in tqdm.tqdm(enumerate(zip_longest(*file_handles))):
                if line_idx % in_memory_lines == 0:
                    if line_idx > 0:  # dumping into tmp files
                        dump_data(prediction_jsonl_files, data, map_to_future, update_fn)
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
                    gt_answer = line_dict["expected_answer"]
                    data[-1][-1] = json.dumps(line_dict)

                    predicted_answer = extract_answer(line_dict)
                    if (predicted_answer, gt_answer) in map_to_future:
                        continue

                    if ignore_cache or line_dict.get("is_correct") is None:
                        map_to_future[(predicted_answer, gt_answer)] = executor.submit(
                            self.is_output_correct,
                            predicted_answer,
                            gt_answer,
                            include_percentage=include_percentage,
                            tolerance=tolerance,
                            timeout=timeout,
                        )
                    else:
                        map_to_future[(predicted_answer, gt_answer)] = DummyFuture(line_dict["is_correct"])

            for file_handle in file_handles:
                file_handle.close()

            if len(data) > 0:
                dump_data(prediction_jsonl_files, data, map_to_future, update_fn)

        write_tmp_files_back(prediction_jsonl_files)


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self):
        return f"http://{self.host}:{self.port}/execute"

    def _parse_request_output(self, output):
        return output.json()

    def _prepare_request(self, generated_code, timeout):
        return {
            "generated_code": generated_code,
            "timeout": timeout,
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


def get_sandbox(sandbox_type, **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f'\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}'
    return python_doc_to_cmd_help(Sandbox, docs_prefix=prefix, arg_prefix="sandbox.")
