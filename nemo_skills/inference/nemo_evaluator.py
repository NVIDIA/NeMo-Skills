# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from nemo_skills.inference.generate import GenerationTask


@dataclass(kw_only=True)
class NemoEvaluatorConfig:
    # Minimal knobs specific to evaluator
    nemo_eval_config_dir: Optional[str] = None
    nemo_eval_config_name: str = "config"
    stream_subprocess_output: bool = True


class NemoEvaluatorGeneration(GenerationTask):
    @classmethod
    def get_generation_default_args(cls) -> str:
        return ""

    @staticmethod
    def _process_mapping(mapping_toml: Dict[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
        """Local processing of mapping.toml into expected dict structure.
        Mirrors nemo_evaluator_launcher.common.mapping._process_mapping.
        """
        mapping: Dict[tuple[str, str], Dict[str, Any]] = {}
        for harness_name, harness_data in mapping_toml.items():
            if not isinstance(harness_data, dict) or "tasks" not in harness_data:
                # skip unrelated keys
                continue
            tasks_by_endpoint = harness_data["tasks"]
            for endpoint_type, tasks in tasks_by_endpoint.items():
                for task_name, task_data in tasks.items():
                    key = (harness_name, task_name)
                    entry = {
                        "task": task_name,
                        "harness": harness_name,
                        "container": harness_data.get("container"),
                        "endpoint_type": endpoint_type,
                    }
                    if isinstance(task_data, dict):
                        entry.update(task_data)
                    mapping[key] = entry
        return mapping

    @staticmethod
    def _load_mapping(
        latest: bool = False, mapping_toml: Optional[str] = None
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        """Load mapping via launcher if available; fallback to local toml parser when not installed.
        Only the subset used by tests is implemented.
        """
        try:
            from nemo_evaluator_launcher.common.mapping import load_tasks_mapping as _ltm  # type: ignore

            return _ltm(latest=latest, mapping_toml=mapping_toml)
        except Exception:
            # Fallback requires mapping_toml
            if not mapping_toml:
                raise RuntimeError("nemo_evaluator_launcher is not available and mapping_toml was not provided")
            with open(mapping_toml, "rb") as f:
                raw = tomllib.load(f)
            return NemoEvaluatorGeneration._process_mapping(raw)

    @staticmethod
    def _get_task_from_mapping(query: str, mapping: Dict[tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
        """Local version of get_task_from_mapping that supports 'task' or 'harness.task'."""
        num_dots = query.count(".")
        if num_dots == 0:
            matches = [k for k in mapping.keys() if k[1] == query]
            if len(matches) == 1:
                return mapping[matches[0]]
            elif len(matches) > 1:
                fully_qualified = [f"{h}.{t}" for (h, t) in matches]
                raise ValueError(
                    f"there are multiple tasks named {query!r} in the mapping, please select one of {fully_qualified!r}"
                )
            else:
                raise ValueError(f"task {query!r} does not exist in the mapping")
        elif num_dots == 1:
            harness, task = query.split(".")
            key = (harness, task)
            if key in mapping:
                return mapping[key]
            raise ValueError(f"harness.task {query!r} does not exist in the mapping")
        else:
            raise ValueError(
                f"invalid query={query!r} for task mapping, it must contain exactly zero or one '.' character"
            )

    @classmethod
    def get_client_container(
        cls,
        *,
        task_queries: List[str],
        use_latest: bool = False,
        mapping_toml: Optional[str] = None,
    ) -> str:
        mapping = cls._load_mapping(latest=use_latest, mapping_toml=mapping_toml)
        containers = set()
        for q in task_queries:
            task_info = cls._get_task_from_mapping(q, mapping)
            containers.add(task_info.get("container"))
        if len(containers) != 1:
            raise ValueError(f"Tasks map to multiple containers: {sorted(containers)}. Split the run per container.")
        return next(iter(containers))

    @classmethod
    def build_eval_command(
        cls,
        *,
        tasks: List[str],
        nemo_eval_config_dir: Optional[str],
        nemo_eval_config_name: str,
        passthrough_overrides: List[str],
    ) -> str:
        base = "export HYDRA_FULL_ERROR=1 && python -m nemo_skills.inference.nemo_evaluator"
        parts: List[str] = [base]
        if tasks:
            # Pass tasks; the underlying script can interpret this or ignore for now
            parts.append(f"++tasks={','.join(tasks)}")
        if nemo_eval_config_dir:
            parts.append(f"++nemo_eval_config_dir={nemo_eval_config_dir}")
        if nemo_eval_config_name:
            parts.append(f"++nemo_eval_config_name={nemo_eval_config_name}")
        if passthrough_overrides:
            parts.extend(passthrough_overrides)
        return " ".join(parts)

    def __init__(self, cfg: NemoEvaluatorConfig | None = None):
        # Skip LLM/model initialization; this class acts as a thin wrapper
        self.cfg = cfg or NemoEvaluatorConfig()

    def generate(self):  # pragma: no cover - end-to-end wiring will be tested separately later
        # Placeholder: execute an evaluator command if provided via cfg; otherwise, no-op
        cmd = self.build_eval_command(
            tasks=[],
            nemo_eval_config_dir=self.cfg.nemo_eval_config_dir,
            nemo_eval_config_name=self.cfg.nemo_eval_config_name,
            passthrough_overrides=[],
        )
        try:
            # Using a no-op echo to validate command format without side effects
            _ = subprocess.list2cmdline(cmd.split())
        except Exception as e:
            raise RuntimeError(f"Failed to build evaluator command: {e}")


# Keep symbol for pipeline dynamic import compatibility if needed
GENERATION_TASK_CLASS = NemoEvaluatorGeneration
