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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from pathlib import Path

import hydra

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

        self.output_dir = Path(self.cfg.output_file).parent

    def log_example_prompt(self, data):
        """SweBench is multi-call benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        return

    def generate_single_answer(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""

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
            f"    --agent.model.name hosted_vllm/SWE-bench/SWE-agent-LM-32B "  # TODO
            f"    --agent.model.api_base http://127.0.0.1:5000/v1 "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # move trajectories to the mounted directory
            f"cp -r trajectories /trajectories_mount/"
        )

        container_name = data_point["container_formatter"].format(
            instance_id=data_point['instance_id'].replace('__', '_1776_')
        )

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f" {container_name} bash -c {shlex.quote(swe_agent_cmd)}"
        )

        # Retry apptainer command up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # no timeout, can work as long as needed
                result = subprocess.run(apptainer_cmd, shell=True, capture_output=True, text=True, timeout=100000)

                # Look for the pred file in the temp directory
                search_path = os.path.join(self.output_dir / "trajectories", "**", f"{data_point['instance_id']}.pred")
                pred_files = glob.glob(search_path, recursive=True)

                if len(pred_files) == 1:
                    # Success, break out of retry loop
                    break
                else:
                    raise ValueError(
                        f"Expected exactly one .pred file for {data_point['instance_id']}, "
                        f"found {len(pred_files)}. Searched in {search_path}"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    LOG.warning(
                        "Attempt %d failed for instance %s: %s. Retrying...",
                        attempt + 1,
                        data_point['instance_id'],
                        str(e),
                    )
                    continue
                else:
                    LOG.error("All %d attempts failed for instance %s", max_retries, data_point['instance_id'])
                    LOG.error("Apptainer command failed. Full logs:")
                    LOG.error("STDOUT:")
                    LOG.error(result.stdout if 'result' in locals() else "No output captured")
                    LOG.error("STDERR:")
                    LOG.error(result.stderr if 'result' in locals() else "No error output captured")
                    LOG.error("Return code: %d", result.returncode if 'result' in locals() else "Unknown")
                    raise ValueError(
                        f"Job logs: Expected exactly one .pred file for {data_point['instance_id']}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}. Searched in {search_path}"
                    )

        with open(pred_files[0], 'r') as f:
            trajectory_dict = json.loads(f.read().strip())

        # need to rename .pred to .jsonl
        pred_mounted_path = (
            pred_files[0].replace(str(self.output_dir), "/trajectories_mount").replace('.pred', '.jsonl')
        )
        with open(pred_files[0].replace('.pred', '.jsonl'), 'w') as f:
            f.write(json.dumps(trajectory_dict))

        swe_bench_cmd = (
            # first installing SWE-bench repo
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "cd /root && "
            "git clone https://github.com/Kipok/SWE-bench.git && "
            "cd SWE-bench && "
            "uv venv --python 3.12 venv && "
            "source venv/bin/activate && "
            "uv pip install -e . && "
            # then running the evaluation
            f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
            f"    --predictions_path {pred_mounted_path} "
            f"    --instance_ids {data_point['instance_id']} "
            f"    --run_id eval-outputs "
            f"    --dataset_name princeton-nlp/SWE-bench_Verified "  # TODO
            f"    --split test"  # TODO (write to file)
            # move outputs
            f" && cp -r logs/run_evaluation/eval-outputs /trajectories_mount/"
        )

        container_name = data_point["container_formatter"].format(
            instance_id=data_point['instance_id'].replace('__', '_1776_')
        )

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f" {container_name} bash -c {shlex.quote(swe_bench_cmd)}"
        )

        # Retry apptainer command up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # no timeout, can work as long as needed
                result = subprocess.run(apptainer_cmd, shell=True, capture_output=True, text=True, timeout=100000)

                # Look for the pred file in the temp directory
                search_path = os.path.join(
                    self.output_dir, "eval-outputs", "**", f"{data_point['instance_id']}/report.json"
                )
                pred_files = glob.glob(search_path, recursive=True)

                if len(pred_files) == 1:
                    # Success, break out of retry loop
                    break
                else:
                    raise ValueError(
                        f"Expected exactly one report.json file for {data_point['instance_id']}, "
                        f"found {len(pred_files)}. Searched in {search_path}"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    LOG.warning(
                        "Attempt %d failed for instance %s: %s. Retrying...",
                        attempt + 1,
                        data_point['instance_id'],
                        str(e),
                    )
                    continue
                else:
                    LOG.error("All %d attempts failed for instance %s", max_retries, data_point['instance_id'])
                    LOG.error("Apptainer command failed. Full logs:")
                    LOG.error("STDOUT:")
                    LOG.error(result.stdout if 'result' in locals() else "No output captured")
                    LOG.error("STDERR:")
                    LOG.error(result.stderr if 'result' in locals() else "No error output captured")
                    LOG.error("Return code: %d", result.returncode if 'result' in locals() else "Unknown")
                    raise ValueError(
                        f"Job logs: Expected exactly one report.json file for {data_point['instance_id']}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}. Searched in {search_path}"
                    )

        with open(pred_files[0], 'r') as f:
            report_json = json.loads(f.read().strip())

        output_dict = {
            "swe-bench-metrics": report_json[data_point['instance_id']],
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO?
        }

        return output_dict

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
