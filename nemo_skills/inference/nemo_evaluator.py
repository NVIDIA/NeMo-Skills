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


import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import OmegaConf

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass(kw_only=True)
class NemoEvaluatorConfig:
    # Minimal knobs specific to evaluator
    nemo_eval_config_dir: Optional[str] = None
    nemo_eval_config_name: str = "config"
    stream_subprocess_output: bool = True
    # Tasks
    tasks: Optional[List[str]] = None


class NemoEvaluatorGeneration(GenerationTask):
    @classmethod
    def get_generation_default_args(cls) -> str:
        return ""

    @staticmethod
    def _load_mapping() -> Dict[tuple[str, str], Dict[str, Any]]:
        """Load mapping via nemo_evaluator_launcher (required)."""
        try:
            from nemo_evaluator_launcher.common.mapping import load_tasks_mapping as _ltm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "nemo_evaluator_launcher is required for evaluator task mapping. "
                "Install with: pip install nemo-evaluator-launcher"
            ) from e
        return _ltm(latest=False)

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
    ) -> str:
        mapping = cls._load_mapping()
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

    # Testable hooks for launcher integration
    @staticmethod
    def build_run_config(config_dir: str, config_name: str, overrides: Optional[List[str]] = None):
        try:
            from nemo_evaluator_launcher.api import RunConfig  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "nemo_evaluator_launcher is required. Install with: pip install nemo-evaluator-launcher"
            ) from e
        return RunConfig.from_hydra(
            config_dir=config_dir,
            config_name=config_name,
            hydra_overrides=overrides or [],
        )

    @staticmethod
    def resolve_task_def(mapping: Dict[tuple[str, str], Dict[str, Any]], task_query: str) -> Dict[str, Any]:
        try:
            from nemo_evaluator_launcher.common.mapping import get_task_from_mapping  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "nemo_evaluator_launcher is required for evaluator task mapping. "
                "Install with: pip install nemo-evaluator-launcher"
            ) from e
        return get_task_from_mapping(task_query, mapping)

    @staticmethod
    def build_task_command(run_cfg, task_cfg, task_def) -> tuple[str, str]:
        try:
            from nemo_evaluator_launcher.common.helpers import get_eval_factory_command  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "nemo_evaluator_launcher is required. Install with: pip install nemo-evaluator-launcher"
            ) from e
        cmd_struct = get_eval_factory_command(run_cfg, task_cfg, task_def)
        return cmd_struct.cmd, cmd_struct.debug

    @staticmethod
    def execute_shell_command(cmd: str, stream: bool) -> int:
        stdout = subprocess.PIPE if stream else subprocess.DEVNULL
        process = subprocess.Popen(cmd, shell=True, stdout=stdout, text=True, bufsize=1)
        if stream and process.stdout is not None:
            for line in process.stdout:
                print(line.rstrip())
            process.stdout.close()
        process.wait()
        return process.returncode

    def __init__(self, cfg: NemoEvaluatorConfig | None = None):
        self.cfg = cfg or NemoEvaluatorConfig()

    def generate(self):
        if not self.cfg.nemo_eval_config_dir:
            raise ValueError("nemo_eval_config_dir is required to build evaluator RunConfig")

        # Build launcher RunConfig
        run_cfg = self.build_run_config(
            config_dir=self.cfg.nemo_eval_config_dir,
            config_name=self.cfg.nemo_eval_config_name,
            overrides=[],
        )

        # Determine tasks to run
        requested_tasks = self.cfg.tasks
        tasks_to_run: List[str]
        if requested_tasks:
            tasks_to_run = requested_tasks
        else:
            # Use tasks from the Hydra config if none explicitly requested
            tasks_to_run = [getattr(t, "name", None) for t in getattr(run_cfg.evaluation, "tasks", [])]
            tasks_to_run = [t for t in tasks_to_run if t]
        LOG.info(f"WIPP req tasks {requested_tasks}")
        if not tasks_to_run:
            LOG.warning("No tasks requested or found in config; nothing to run")
            return

        # Load mapping and run each task sequentially
        mapping = self._load_mapping()

        name_to_task_cfg = {getattr(t, "name", None): t for t in getattr(run_cfg.evaluation, "tasks", [])}
        for task_name in tasks_to_run:
            task_cfg = name_to_task_cfg.get(task_name)
            if task_cfg is None:
                LOG.warning(
                    "Task %s not present in run_cfg.evaluation.tasks; proceeding with default task cfg", task_name
                )
                # Some launchers may allow constructing a default task_cfg shape; here we require presence
                # Continue to next task rather than fail-hard
                continue

            task_def = self.resolve_task_def(mapping, task_name)
            cmd, debug = self.build_task_command(run_cfg, task_cfg, task_def)
            LOG.info("Generated evaluator command", extra={"cmd": cmd, "debug": debug})

            rc = self.execute_shell_command(cmd, self.cfg.stream_subprocess_output)
            if rc != 0:
                raise RuntimeError(f"Evaluator for task {task_name} exited with code {rc}")


# Keep symbol for pipeline dynamic import compatibility if needed
GENERATION_TASK_CLASS = NemoEvaluatorGeneration


# Hydra entrypoint to allow `python -m nemo_skills.inference.nemo_evaluator ++overrides`
@hydra.main(version_base=None, config_name=None)
def main(cfg):  # cfg is an OmegaConf DictConfig built from ++ overrides
    setup_logging()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    # Normalize tasks: accept comma-delimited string or list
    tasks_val = cfg_dict.get("tasks")
    if isinstance(tasks_val, str):
        tasks_val = [t.strip() for t in tasks_val.split(",") if t.strip()]
    evaluator_cfg = NemoEvaluatorConfig(
        nemo_eval_config_dir=cfg_dict.get("nemo_eval_config_dir"),
        nemo_eval_config_name=str(cfg_dict.get("nemo_eval_config_name", "config")),
        stream_subprocess_output=bool(cfg_dict.get("stream_subprocess_output", True)),
        tasks=tasks_val,
    )
    task = NemoEvaluatorGeneration(evaluator_cfg)
    task.generate()


if __name__ == "__main__":
    main()
