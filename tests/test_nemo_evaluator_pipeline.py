# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import importlib
import pytest


def _import_pipeline_cmd():
    try:
        mod = importlib.import_module("nemo_skills.pipeline.nemo_evaluator")
    except Exception:
        pytest.skip("nemo_skills.pipeline.nemo_evaluator not implemented yet")
    return mod


def test_cli_parsing_and_dry_run(monkeypatch, tmp_path):
    mod = _import_pipeline_cmd()
    app = getattr(mod, "app", None)
    command_fn = getattr(mod, "nemo_evaluator", None)
    if app is None or command_fn is None:
        pytest.skip("nemo_evaluator Typer command not implemented yet")

    # Simulate Typer call via function directly; ensure no exceptions for dry_run
    class Ctx:
        args = [
            "++nemo_eval_config_dir=/configs/evaluator",
            "++nemo_eval_config_name=config",
        ]

    # Provide minimal valid args
    kwargs = dict(
        ctx=Ctx(),
        cluster=None,
        output_dir=str(tmp_path / "out"),
        expname="evaluator-test",
        tasks="aime_2025_nemo",
        job_gpus=0,
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
        latest_mapping=False,
        tasks_mapping_toml=None,
    )

    # Should not raise; actual Pipeline.run is expected to handle dry_run path
    try:
        command_fn(**kwargs)
    except Exception as e:
        pytest.fail(f"CLI dry_run failed unexpectedly: {e}")
