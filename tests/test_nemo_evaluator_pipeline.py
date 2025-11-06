# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import importlib
import textwrap

import pytest


def _import_pipeline_cmd():
    try:
        mod = importlib.import_module("nemo_skills.pipeline.nemo_evaluator")
    except Exception:
        pytest.skip("nemo_skills.pipeline.nemo_evaluator not implemented yet")
    return mod


def test_cli_parsing_and_dry_run(monkeypatch, tmp_path):
    pytest.importorskip("nemo_evaluator_launcher")
    mod = _import_pipeline_cmd()
    app = getattr(mod, "app", None)
    command_fn = getattr(mod, "nemo_evaluator", None)
    if app is None or command_fn is None:
        pytest.skip("nemo_evaluator Typer command not implemented yet")

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
        command_fn(**kwargs)
    except Exception as e:
        pytest.fail(f"CLI dry_run failed unexpectedly: {e}")
