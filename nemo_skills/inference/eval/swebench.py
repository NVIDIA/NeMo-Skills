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

from nemo_skills.inference.generate import GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


# not inheriting since most parameters are not supported because we don't use our model client here
# TODO: should we fix that?
@nested_dataclass(kw_only=True)
class SweBenchGenerationConfig:
    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations

    # SWE-agent configuration file path. Can be specified in the same way as ns prompt configs
    # TODO: that's probably not a good default, right?
    sweagent_config: str = "eval/swe-bench/swe-agent"
    swebench_tests_timeout: int = 60 * 30  # Timeout for the tests after applying the patch, in seconds

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # maximum number of concurrent requests to the server for the async loop
    # if sync loop is used, this is the batch size
    max_concurrent_requests: int = 512
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    # if False, will not add num_generated_tokens and generation_time values.
    # Useful when running judge jobs to keep the original generation statistics
    add_generation_stats: bool = True
    generation_key: str = "generation"
    use_async_loop: bool = True
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict
    dry_run: bool = False

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    remove_thinking: bool = False
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"

    prompt_format: str = "ns"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swebench_generation_config", node=SweBenchGenerationConfig)


class SweBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: SweBenchGenerationConfig):
        # not calling parent init on purpose
        self.cfg = cfg
        self.use_async_loop = True  # SweBench is a multi-call benchmark, so we have to use async loop
        self.output_dir = Path(self.cfg.output_file).parent

        self.executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent_requests)

    def log_example_prompt(self, data):
        """SweBench is multi-call benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        return

    def _execute_container_command(
        self, data_point, command, expected_file_pattern, mode, max_retries=3, timeout=100000
    ):
        """Execute a command in an Apptainer container with retry logic."""
        container_name = data_point["container_formatter"].format(
            instance_id=data_point['instance_id'].replace('__', '_1776_')
        )

        # Create logs directory if it doesn't exist
        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{data_point['instance_id']}_{mode}.log"

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f" {container_name} bash -c {shlex.quote(command)}"
        )

        # Retry apptainer command up to max_retries times
        for attempt in range(max_retries):
            try:
                # Stream output to log file as it appears
                with open(log_file_path, 'w') as log_file:
                    try:
                        result = subprocess.run(
                            apptainer_cmd, shell=True, text=True, timeout=timeout, stdout=log_file, stderr=log_file
                        )
                    except subprocess.TimeoutExpired:
                        attempt = max_retries  # Force exit the loop on timeout
                        raise ValueError()

                # Look for the expected file
                pred_files = glob.glob(expected_file_pattern, recursive=True)

                if len(pred_files) == 1:
                    # Success, break out of retry loop
                    return pred_files[0]
                else:
                    raise ValueError(
                        f"Expected exactly one file matching {expected_file_pattern} for {data_point['instance_id']}, "
                        f"found {len(pred_files)}."
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    LOG.warning(
                        "Attempt %d failed for instance %s. Retrying...",
                        attempt + 1,
                        data_point['instance_id'],
                    )
                    continue
                else:
                    LOG.error("All %d attempts failed for instance %s", max_retries, data_point['instance_id'])
                    LOG.error("Apptainer command failed. Check logs at: %s", log_file_path)
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file_path}. "
                        f"Expected exactly one file matching {expected_file_pattern}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                    )

    def generate_single_answer(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?
        # TODO: we should disallow any non-supported parameters (e.g. top-k or min-p) by checking against defaults
        # TODO: how should we handle tokens_to_generate?
        # TODO: is random_seed different on different reruns? Can we force it to?
        print(f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1")
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
            f"    --config {get_config_path(self.cfg.sweagent_config)} "
            f"    --agent.model.name hosted_vllm/{self.cfg.server.model} "
            f"    --agent.model.api_base http://{self.cfg.server.host}:{self.cfg.server.port}/v1 "
            f"    --agent.model.temperature {self.cfg.inference.temperature} "
            f"    --agent.model.top_p {self.cfg.inference.top_p} "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # move trajectories to the mounted directory
            f"cp -r trajectories /trajectories_mount/"
        )

        # Execute SWE-agent command
        search_path = os.path.join(self.output_dir / "trajectories", "**", f"{data_point['instance_id']}.pred")
        pred_file = self._execute_container_command(data_point, swe_agent_cmd, search_path, mode="agent")

        with open(pred_file, 'r') as f:
            trajectory_dict = json.loads(f.read().strip())

        # need to rename .pred to .jsonl
        pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount").replace('.pred', '.jsonl')
        with open(pred_file.replace('.pred', '.jsonl'), 'w') as f:
            f.write(json.dumps(trajectory_dict))

        # TODO: get num_generated_tokens and other stats from .traj file
        # looks like data['info']['model_stats']
        # {'instance_cost': 0, 'tokens_sent': 40858, 'tokens_received': 1775, 'api_calls': 9}

        # Check if the trajectory has an empty patch before running evaluation
        has_patch = trajectory_dict['model_patch'] is not None

        if not has_patch:
            report_json = {
                data_point['instance_id']: {
                    "resolved": False,
                    "patch_exists": False,
                    "patch_successfully_applied": False,
                }
            }
        else:
            # Run full evaluation with streaming output
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
                # then running the evaluation with streaming output
                f"/root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                f"    --predictions_path {pred_mounted_path} "
                f"    --instance_ids {data_point['instance_id']} "
                f"    --run_id eval-outputs "
                f"    --timeout {self.cfg.swebench_tests_timeout} "
                f"    --dataset_name {data_point['dataset_name']} "
                f"    --split {data_point['split']} && "
                f"cp -r logs/run_evaluation/eval-outputs /trajectories_mount/"
            )

            # Execute SWE-bench evaluation command
            search_path = os.path.join(
                self.output_dir, "eval-outputs", "**", f"{data_point['instance_id']}/report.json"
            )
            # TODO: should we fail on errors here? Seems that json isn't always generated
            try:
                report_file = self._execute_container_command(
                    data_point,
                    swe_bench_cmd,
                    search_path,
                    mode="eval",
                    timeout=self.cfg.swebench_tests_timeout + 120,
                )
            except ValueError:
                LOG.error("Failed to execute SWE-bench evaluation command for %s", data_point['instance_id'])
                report_json = {
                    data_point['instance_id']: {
                        "resolved": False,
                        "patch_exists": True,
                        "patch_successfully_applied": False,
                    }
                }
                report_file = None

            if report_file is not None:
                with open(report_file, 'r') as f:
                    report_json = json.loads(f.read().strip())

        output_dict = {
            "swe-bench-metrics": report_json[data_point['instance_id']],
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }

        return output_dict

    def llm_generate(self, data_points, data, is_async=False):
        futures = []
        for data_point in data_points:
            future = self.executor.submit(self.generate_single_answer, data_point, data)
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
