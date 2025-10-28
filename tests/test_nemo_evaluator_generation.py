# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import importlib
import textwrap
from collections import defaultdict

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
    pytest.importorskip("nemo_evaluator_launcher")
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")
    # Use latest mapping from launcher and pick any task
    from nemo_evaluator_launcher.common.mapping import load_tasks_mapping

    mp = load_tasks_mapping(latest=True)
    (harness, task), info = next(iter(mp.items()))
    fq = f"{harness}.{task}"
    container = get_client_container(task_queries=[fq])
    assert container == info.get("container")


def test_get_client_container_multi_same_container(sample_mapping_toml):
    pytest.importorskip("nemo_evaluator_launcher")
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")
    from nemo_evaluator_launcher.common.mapping import load_tasks_mapping

    mp = load_tasks_mapping(latest=True)
    by_container = defaultdict(list)
    for (h, t), info in mp.items():
        by_container[info.get("container")].append((h, t))
    pair = next(((c, tasks) for c, tasks in by_container.items() if len(tasks) >= 2), None)
    if not pair:
        pytest.skip("No two tasks with same container found in current mapping")
    container_expect, tasks_list = pair
    q = [f"{h}.{t}" for h, t in tasks_list[:2]]
    container = get_client_container(task_queries=q)
    assert container == container_expect


def test_get_client_container_conflict(sample_mapping_toml):
    pytest.importorskip("nemo_evaluator_launcher")
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")
    from nemo_evaluator_launcher.common.mapping import load_tasks_mapping

    mp = load_tasks_mapping(latest=True)
    items = list(mp.items())
    # Find two tasks with different containers
    first_h, first_t = items[0][0]
    first_c = items[0][1].get("container")
    second = next(((h, t, info) for (h, t), info in items[1:] if info.get("container") != first_c), None)
    if second is None:
        pytest.skip("No conflict containers available in mapping")
    second_h, second_t, _ = second
    with pytest.raises(Exception):
        get_client_container(task_queries=[f"{first_h}.{first_t}", f"{second_h}.{second_t}"])


def test_get_client_container_ambiguous(sample_mapping_toml):
    pytest.importorskip("nemo_evaluator_launcher")
    Gen = _import_generation_class()
    get_client_container = getattr(Gen, "get_client_container", None)
    if get_client_container is None:
        pytest.skip("get_client_container not implemented yet")
    from nemo_evaluator_launcher.common.mapping import load_tasks_mapping

    mp = load_tasks_mapping(latest=True)
    task_to_harnesses = defaultdict(list)
    for (h, t) in mp.keys():
        task_to_harnesses[t].append(h)
    ambiguous = next((t for t, hs in task_to_harnesses.items() if len(hs) >= 2), None)
    if not ambiguous:
        pytest.skip("No ambiguous task names found in current mapping")
    with pytest.raises(Exception):
        get_client_container(task_queries=[ambiguous])


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
