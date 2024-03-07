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
from typing import Dict, List, Optional, Tuple

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
    NOT_EXECUTED = "<not_executed>"
    EXECUTION_ERROR = "Execution error:"
    SYNTAX_ERROR = "Syntax error:"
    RESULT_NOT_DEFINED_ERROR = "Result is not defined"
    TIMEOUT_ERROR = "timeout"
    UNDEFINED_ERROR = "Undefined error:"
    ERROR_PREFIXES = (EXECUTION_ERROR, SYNTAX_ERROR, RESULT_NOT_DEFINED_ERROR, TIMEOUT_ERROR, UNDEFINED_ERROR)

    @abc.abstractmethod
    def execute_code(self, generated_code, timeout=10.0, max_output_characters=1000) -> Dict:
        pass

    @abc.abstractmethod
    def is_output_correct(self, pred_output, gt_output, timeout=10.0) -> bool:
        pass

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

                    predicted_answer = line_dict.get("predicted_answer", Sandbox.NOT_EXECUTED)

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
    """Locally hosted sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: str = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: str = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server=None,
        ssh_key_path=None,
    ):
        self.host = host
        self.port = port
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        # will keep state of code sessions
        self.sessions = {}

    def clear_session(self, session_id):
        del self.sessions[session_id]

    def _send_request(self, request, timeout, endpoint):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            output = sshtunnel_request.put(
                url=f"http://{self.host}:{self.port}/{endpoint}",
                data=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        else:
            import requests

            output = requests.put(
                url=f"http://{self.host}:{self.port}/{endpoint}",
                data=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )

        return output

    def execute_code(
        self,
        generated_code: str,
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict, str]:
        import requests

        if session_id is None:  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []

        request = {
            "generated_code": generated_code,
            "timeout": timeout,
            "max_output_characters": max_output_characters,
            "state": self.sessions[session_id],
        }
        try:
            output = self._send_request(request, timeout, "execute_code").json()
            state = output.pop("state")
            if state is not None:
                self.sessions[session_id] = state
        except requests.exceptions.Timeout:
            output = {'result': None, 'error_message': Sandbox.TIMEOUT_ERROR}
        return output, session_id

    def is_output_correct(self, pred_output, gt_output, include_percentage=True, tolerance=1e-4, timeout=10.0):
        import requests

        request = {
            "pred_output": pred_output,
            "gt_output": gt_output,
            "include_percentage": include_percentage,
            "tolerance": tolerance,
            "timeout": timeout,
        }
        try:
            output = self._send_request(request, timeout, "is_output_correct").json()
        except requests.exceptions.Timeout:
            output = False
        return output


sandboxes = {
    'local': LocalSandbox,
}


def get_sandbox(sandbox_type, **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f'\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}'
    # only exposing docs for local sandbox for now. Need to change when we support other types
    return python_doc_to_cmd_help(LocalSandbox, docs_prefix=prefix, arg_prefix="sandbox.")
