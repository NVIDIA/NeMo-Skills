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

import os
import shutil
import subprocess


def test_metrics(tmp_path):
    """Current test is very strict and expects the output to match exactly.

    Ideally we should relax that, but keeping like this for now.
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
        ["ns", "summarize_results", str(dst)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ns summarize_results failed: {result.stderr}"

    # 4. Compare output (excluding last line) to expected output file
    output_lines = result.stdout.rstrip('\n').split('\n')
    output_without_last = '\n'.join(output_lines[:-1]) + '\n' if len(output_lines) > 1 else ''
    expected_path = os.path.join(os.path.dirname(__file__), "data/eval_outputs/summarize_results_output.txt")
    with open(expected_path, "r") as f:
        expected = f.read()
    assert output_without_last == expected, "summarize_results output does not match expected output"

    # 5. Check that metrics.json matches metrics.json-test
    metrics_path = dst / "metrics.json"
    metrics_ref_path = os.path.join(os.path.dirname(__file__), "data/eval_outputs/eval-results/metrics.json-test")
    with open(metrics_path, "r") as f:
        metrics = f.read()
    with open(metrics_ref_path, "r") as f:
        metrics_ref = f.read()
    assert metrics == metrics_ref, "metrics.json does not match metrics.json-test"
