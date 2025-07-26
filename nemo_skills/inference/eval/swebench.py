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

import glob
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SweBenchGenerationConfig(GenerateSolutionsConfig):
    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "eval/swe-bench/swe-agent"
    trajectories_dir: str = "/tmp/swe_trajectories"

    # TODO: disallow most parameters?
    # TODO: can we support our model interface?


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swebench_generation_config", node=SweBenchGenerationConfig)


class SweBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: SweBenchGenerationConfig):
        super().__init__(cfg)

        if not self.use_async_loop:  # if it was True, this message is printed by base class
            LOG.info(
                "Async loop is maintaining %d generations in parallel. "
                "Use max_concurrent_requests to control the number of concurrent requests.",
                self.cfg.max_concurrent_requests,
            )
            if self.server["server_type"] in ["nemo", "megatron"] and self.prompt_template is None:
                LOG.warning(
                    "NeMo/Megatron servers don't support inflight batching, "
                    "but SweBench evaluation requires it for efficient inference. "
                    "Each request will be processed 1 by 1, which is extremely inefficient and slow! "
                    "We highly recommend switching to a server that supports inflight batching."
                )
        self.use_async_loop = True  # SweBench is a multi-call benchmark, so we have to use async loop

        # self.container_ports = os.environ["NEMO_SKILLS_SANDBOX_PORTS"].split(',')
        # self.sandboxes = [Sandbox(port=port) for port in self.container_ports]

    def log_example_prompt(self, data):
        """SweBench is multi-call benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        return

    def generate_single_answer(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""

        # Create temporary directory for trajectories
        os.makedirs(self.cfg.trajectories_dir, exist_ok=True)

        swe_agent_cmd = (
            # first installing swe-agent repo
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "cd /root && "
            "git clone https://github.com/SWE-agent/SWE-agent.git && "
            "cd SWE-agent && "
            "uv venv --python 3.12 venv && "
            "source venv/bin/activate && "
            "uv pip install -e . && "
            # then running the agent
            f"/root/SWE-agent/venv/bin/python -m sweagent run "
            f"    --config /nemo_run/code/nemo_skills/prompt/config/{self.cfg.prompt_config}.yaml "  # TODO: handle absolute path!
            f"    --agent.model.name self_hosted_model "
            f"    --agent.model.api_base http://127.0.0.1:5000/v1 "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # move trajectories to the mounted directory
            f"mv trajectories/* /trajectories_mount/"
        )

        container_name = data_point["container_formatter"].format(
            instance_id=data_point['instance_id'].replace('__', '_1776_')
        )

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.cfg.trajectories_dir},dst=/trajectories_mount "
            f" docker://{container_name} bash -c {shlex.quote(swe_agent_cmd)}"
        )

        LOG.info("Running command: %s", apptainer_cmd)

        # no timeout, can work as long as needed
        subprocess.run(apptainer_cmd, shell=True, capture_output=True, text=True, timeout=100000)

        # Look for the pred file in the temp directory
        search_path = os.path.join(self.cfg.trajectories_dir, "**", f"{data_point['instance_id']}.pred")
        pred_files = glob.glob(search_path, recursive=True)

        if len(pred_files) != 1:
            raise ValueError(
                f"Expected exactly one .pred file for {data_point['instance_id']}, "
                f"found {len(pred_files)}. Searched in {search_path}"
            )
        with open(pred_files[0], 'r') as f:
            trajectory_json = f.read().strip()

        return {'generation': trajectory_json}

    def llm_generate(self, data_points, data, is_async=False):
        futures = []

        with ThreadPoolExecutor(max_workers=len(data_points)) as executor:
            for data_point in data_points:
                future = executor.submit(self.generate_single_answer, data_point, data)
                futures.append(future)

        return futures

    def get_llm_generations(self, requests_in_progress, generations):
        for dp_idx, future in requests_in_progress.items():
            if future.done():
                generations[dp_idx] = future.result()
            else:
                generations[dp_idx] = {'generation': None}

        return requests_in_progress, generations


GENERATION_TASK_CLASS = SweBenchGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_swebench_generation_config')
def swebench_generation(cfg: SweBenchGenerationConfig):
    cfg = SweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SweBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SweBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swebench_generation()
