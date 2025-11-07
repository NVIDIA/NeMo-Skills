# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import textwrap

import pytest

from nemo_skills.pipeline.nemo_evaluator import nemo_evaluator as nemo_evaluator_fn


def test_cli_parsing_and_dry_run(monkeypatch, tmp_path):
    # Simulate Typer call via function directly; ensure no exceptions for dry_run
    class Ctx:
        args = []

    # Provide minimal valid args
    kwargs = dict(
        ctx=Ctx(),
        cluster=None,
        output_dir=str(tmp_path / "out"),
        expname="evaluator-test",
        nemo_evaluator_config=str(tmp_path / "example-eval-config.yaml"),
        job_nodes=1,
        partition=None,
        qos=None,
        time_min=None,
        mount_paths=None,
        log_dir=None,
        exclusive=False,
        with_sandbox=False,
        keep_mounts_for_sandbox=False,
        reuse_code=True,
        reuse_code_exp=None,
        run_after=None,
        dependent_jobs=0,
        dry_run=True,
    )

    # Should not raise; actual Pipeline.run is expected to handle dry_run path
    try:
        # write a tiny config file for the launcher
        (tmp_path / "example-eval-config.yaml").write_text(
            textwrap.dedent(
                """
                defaults:
                  - execution: local
                  - deployment: none
                  - _self_

                execution:
                  output_dir: test

                target:
                  api_endpoint:
                    model_id: meta/llama-3.1-8b-instruct
                    url: http://127.0.0.1:8000/v1/chat/completions

                evaluation:
                  tasks:
                    - name: ifeval
                """
            ).strip()
        )
        nemo_evaluator_fn(**kwargs)
    except Exception as e:
        pytest.fail(f"CLI dry_run failed unexpectedly: {e}")
