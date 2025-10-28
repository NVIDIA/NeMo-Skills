import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_evaluator_launcher")

from nemo_skills.inference.nemo_evaluator import NemoEvaluatorConfig, NemoEvaluatorGeneration


class FakeGen(NemoEvaluatorGeneration):
    @staticmethod
    def build_run_config(config_dir, config_name, overrides=None):
        # Provide a minimal object with evaluation.tasks having names
        return SimpleNamespace(
            evaluation=SimpleNamespace(tasks=[SimpleNamespace(name="ifeval")])
        )

    @staticmethod
    def build_task_command(run_cfg, task_cfg, task_def):
        # Return an echo command that is safe
        return ("bash -lc 'echo OK'", "debug")


def test_generate_single_task_stream_true(capsys):
    cfg = NemoEvaluatorConfig(
        nemo_eval_config_dir="tests/data/nemo_evaluator",
        nemo_eval_config_name="example-eval-config",
        tasks=["ifeval"],
        stream_subprocess_output=True,
    )
    FakeGen(cfg).generate()
    out = capsys.readouterr().out
    assert "OK" in out


def test_generate_single_task_stream_false(capsys):
    cfg = NemoEvaluatorConfig(
        nemo_eval_config_dir="tests/data/nemo_evaluator",
        nemo_eval_config_name="example-eval-config",
        tasks=["ifeval"],
        stream_subprocess_output=False,
    )
    FakeGen(cfg).generate()
    out = capsys.readouterr().out
    assert "OK" not in out  # no streaming


def test_generate_failure_raises(monkeypatch):
    class FailGen(NemoEvaluatorGeneration):
        @staticmethod
        def build_run_config(config_dir, config_name, overrides=None):
            return SimpleNamespace(
                evaluation=SimpleNamespace(tasks=[SimpleNamespace(name="ifeval"), SimpleNamespace(name="ifeval2")])
            )

        @staticmethod
        def build_task_command(run_cfg, task_cfg, task_def):
            # First command fails, second would echo if reached
            if task_cfg.name == "ifeval":
                return ("bash -lc 'exit 7'", "debug")
            return ("bash -lc 'echo SHOULD_NOT_RUN'", "debug")

    cfg = NemoEvaluatorConfig(
        nemo_eval_config_dir="tests/data/nemo_evaluator",
        nemo_eval_config_name="example-eval-config",
        tasks=["ifeval", "ifeval2"],
        stream_subprocess_output=True,
    )
    with pytest.raises(RuntimeError):
        FailGen(cfg).generate()
