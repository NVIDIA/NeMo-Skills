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

import json
import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("max_seq_len", [None, 8192, 32768])
def test_metrics(tmp_path, max_seq_len):
    """
    Test that the reference metrics.json file matches the generated one, and also tests that the ns summarize_results command runs successfully.

    To update the expected output do the following:
    1. Run the test with `pytest tests/test_metrics.py -s`. It will print the tmp_path it's using.
    2. Replace the metrics.json file with the generated one:
       `cp <tmp_path>/eval-results/metrics.json tests/data/eval_outputs/eval-results/metrics.json-test`
    """
    # 1. Copy eval-results to tmp_path
    src = os.path.join(os.path.dirname(__file__), "data/eval_outputs/eval-results")
    dst = tmp_path / "eval-results"
    shutil.copytree(src, dst)

    # 2. Recursively rename .jsonl-test files to .jsonl
    for root, _, files in os.walk(dst):
        for fname in files:
            if fname.endswith(".jsonl-test"):
                old_path = os.path.join(root, fname)
                new_path = os.path.join(root, fname.replace(".jsonl-test", ".jsonl"))
                os.rename(old_path, new_path)

    # 3. Run ns summarize_results {tmp_path}
    result = subprocess.run(
        ["ns", "summarize_results", str(dst)] + ([f"--max_seq_len={max_seq_len}"] if max_seq_len else []),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ns summarize_results failed: {result.stderr}"

    # 4. Check that metrics.json matches metrics.json-test
    ref_suffix = "" if max_seq_len in [None, 32768] else f"-ms{max_seq_len}"
    metrics_path = dst / "metrics.json"
    metrics_ref_path = os.path.join(
        os.path.dirname(__file__), f"data/eval_outputs/eval-results/metrics{ref_suffix}.json-test"
    )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(metrics_ref_path, "r") as f:
        metrics_ref = json.load(f)

    def recursive_compare_metrics(metrics1, metrics2, tol=1e-6):
        """Compare numerical values in nested JSON structures of metrics."""

        if type(metrics1) != type(metrics2):
            return False

        if isinstance(metrics1, dict):
            return metrics1.keys() == metrics2.keys() and all(
                recursive_compare_metrics(metrics1[key], metrics2[key], tol) for key in metrics1
            )

        if isinstance(metrics1, (int, float)) and isinstance(metrics2, (int, float)):
            return abs(metrics1 - metrics2) <= tol, f"Metrics do not match: {metrics1} != {metrics2}"

        if isinstance(metrics1, list):
            return len(metrics1) == len(metrics2) and all(
                recursive_compare_metrics(a, b, tol) for a, b in zip(metrics1, metrics2)
            )

        return metrics1 == metrics2

    assert recursive_compare_metrics(metrics, metrics_ref) is True, "Metrics do not match"
