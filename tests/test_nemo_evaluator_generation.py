# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import importlib
import textwrap

import pytest


@pytest.fixture()
def sample_mapping_toml(tmp_path):
    """Create a minimal mapping.toml compatible with nemo_evaluator_launcher mapping loader."""
    content = textwrap.dedent(
        """
        [math_harness]
        container = "nvcr.io/nvidia/eval-factory/simple-evals:25.08.1"
        [math_harness.tasks]
        [math_harness.tasks.chat]
        aime_2025_nemo = { difficulty = "easy" }

        [code_harness]
        container = "nvcr.io/nvidia/code-evals:1.0.0"
        [code_harness.tasks]
        [code_harness.tasks.chat]
        humaneval = { split = "test" }

        [other_harness]
        container = "nvcr.io/nvidia/eval-factory/simple-evals:25.08.1"
        [other_harness.tasks]
        [other_harness.tasks.chat]
        aime_2025_nemo = { difficulty = "hard" }
        """
    ).strip()
    p = tmp_path / "mapping.toml"
    p.write_text(content)
    return str(p)


def _import_generation_class():
    try:
        mod = importlib.import_module("nemo_skills.inference.nemo_evaluator")
    except Exception:
        pytest.skip("nemo_skills.inference.nemo_evaluator not implemented yet")
    gen = getattr(mod, "NemoEvaluatorGeneration", None)
    if gen is None:
        pytest.skip("NemoEvaluatorGeneration class not found (implementation pending)")
    return gen


def test_get_client_container_single_task(sample_mapping_toml):
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")

    container = get_client_container(
        task_queries=["aime_2025_nemo"], use_latest=False, mapping_toml=sample_mapping_toml
    )
    assert container == "nvcr.io/nvidia/eval-factory/simple-evals:25.08.1"


def test_get_client_container_multi_same_container(sample_mapping_toml):
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")

    # Both tasks do not exist under same harness; keep single existing task twice to verify set logic
    container = get_client_container(
        task_queries=["aime_2025_nemo", "math_harness.aime_2025_nemo"],
        use_latest=False,
        mapping_toml=sample_mapping_toml,
    )
    assert container == "nvcr.io/nvidia/eval-factory/simple-evals:25.08.1"


def test_get_client_container_conflict(sample_mapping_toml):
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")

    with pytest.raises(Exception):
        get_client_container(
            task_queries=["aime_2025_nemo", "humaneval"], use_latest=False, mapping_toml=sample_mapping_toml
        )


def test_get_client_container_ambiguous(sample_mapping_toml):
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")

    # Query without harness when multiple harnesses contain same task must error
    with pytest.raises(Exception):
        get_client_container(task_queries=["aime_2025_nemo"], use_latest=False, mapping_toml=sample_mapping_toml)


def test_build_eval_command_includes_overrides(sample_mapping_toml):
    Gen = _import_generation_class()
    build_eval_command = getattr(Gen, "build_eval_command", None)
    if build_eval_command is None:
        pytest.skip("build_eval_command not implemented yet")

    cmd = build_eval_command(
        tasks=["aime_2025_nemo"],
        nemo_eval_config_dir="/configs/evaluator",
        nemo_eval_config_name="config",
        passthrough_overrides=["++foo=bar", "++x.y=1"],
    )
    # Basic assertions; do not over-specify exact formatting
    assert "python -m" in cmd
    assert "nemo_evaluator" in cmd  # module path contains evaluator
    assert "++nemo_eval_config_dir=/configs/evaluator" in cmd
    assert "++nemo_eval_config_name=config" in cmd
    assert "++foo=bar" in cmd and "++x.y=1" in cmd
