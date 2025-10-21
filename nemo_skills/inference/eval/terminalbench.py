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

import asyncio
import json
import logging
import shlex
import sys
from dataclasses import field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# path to clone terminal-bench repo inside nemo-skills container
TB_REPO_PATH = "/root/tb"


# Like nemo_skills.inference.generate.InferenceConfig, except most parameters are not passed by default
# because they may not be supported by all LLM servers.
# TODO: how do you pass these to tb run? if it's not supported, then we should implement it in the fork
@nested_dataclass(kw_only=True)
class TerminalBenchInferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int | None = None
    top_p: float = 0.95
    min_p: float | None = None
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None


# not inheriting since most parameters are not supported because we don't use our model client here
# TODO: should we fix that?
@nested_dataclass(kw_only=True)
class TerminalBenchGenerationConfig:
    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations

    agent: str  # Which agent to use. TODO: set a default for this maybe?

    # URL of the Terminal-Bench repo to pass to git clone. TODO: set default to public tb fork once we have one
    tb_repo: str
    tb_commit: str = "HEAD"  # Which commit to use when cloning the Terminal-Bench repo

    # LLM call parameters
    inference: TerminalBenchInferenceConfig = field(default_factory=TerminalBenchInferenceConfig)
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
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict
    dry_run: bool = False

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    remove_thinking: bool = False
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_terminalbench_generation_config", node=TerminalBenchGenerationConfig)


class TerminalBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: TerminalBenchGenerationConfig):
        self.cfg = cfg

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )
        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

        # needs to skip completed samples, not used otherwise
        self.cfg.prompt_format = "ns"

        # set up output folder,
        # making sure it is different for each random seed if we're running with --benchmarks=terminal-bench:N
        # to avoid overwriting files
        self.output_dir = Path(self.cfg.output_file).parent
        if self.cfg.inference.random_seed is not None:
            self.output_dir = self.output_dir / f"rs{self.cfg.inference.random_seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # set up terminal-bench repo
        setup_cmd = (
            f"git clone {self.cfg.tb_repo} {TB_REPO_PATH} && "
            f"cd {TB_REPO_PATH} && "
            f"git checkout {self.cfg.tb_commit} && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "uv venv && "
            "source .venv/bin/activate && "
            "uv pip install -e ."
        )
        asyncio.run(self._run_command(setup_cmd, self.output_dir / "setup.log"))

    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        return

    def setup_llm(self):
        return

    def setup_litellm_cache(self):
        return

    def cleanup_litellm_cache(self):
        return

    async def apply_evaluation_hook(self, data_point):
        # currently evaluation is done directly after generation already
        return data_point

    async def _run_command(self, cmd, log_path):
        with open(log_path, "w") as log_file:
            # Create async subprocess
            process = await asyncio.create_subprocess_shell(
                f"/bin/bash -c {shlex.quote(cmd)}", stdout=log_file, stderr=log_file
            )

            # Wait for completion
            await asyncio.wait_for(process.communicate(), timeout=None)

            if process.returncode != 0:
                raise ValueError(f"Command failed with return code {process.returncode}")

    async def process_single_datapoint(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        runs_dir = self.output_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = self.output_dir / "run-logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # TODO: how to handle custom datasets?
        #       currently tasks are stored directly inside the repo as subfolders.
        #       will we need datasets that are stored outside of the tb repo we're cloning?
        #
        # TODO: how to handle containers?
        #       currently the container names are stored in docker-compose.yaml for each task (see nv-internal folder),
        #       and the code handles downloading .sif files or reusing existing ones inside of --images-base-path.
        #       should we reuse that logic, or make the user download and reference the containers like in swe-bench?
        tb_cmd = (
            f"cd {TB_REPO_PATH} && "
            f"source .venv/bin/activate && "
            f"tb run "
            f"    --agent {self.cfg.agent} "
            f"    --model hosted_vllm/{self.cfg.server.model} "
            f"    --agent-kwarg api_base={api_base} "
            f"    --dataset-path nv-internal "
            f"    --task-id {data_point['task_id']} "
            f"    --backend singularity "
            f"    --images-base-path /terminal-bench-images "
            f"    --output-path runs "
            f"    --run-id {data_point['task_id']} && "
            f"cp -r runs/{data_point['task_id']} {runs_dir}"
        )

        await self._run_command(tb_cmd, logs_dir / f"{data_point['task_id']}.log")

        results_path = runs_dir / data_point["task_id"] / "results.json"
        if not results_path.exists():
            raise ValueError(f"Expected a results file at {results_path} but did not find one.")

        results = json.loads(results_path.read_text())
        return {
            "resolved": results["results"][0]["is_resolved"],
            "terminal-bench-results": results,
            "generation": "",  # TODO: what should go here? the benchmark is multiturn
        }


GENERATION_TASK_CLASS = TerminalBenchGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_terminalbench_generation_config")
def terminalbench_generation(cfg: TerminalBenchGenerationConfig):
    cfg = TerminalBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = TerminalBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    TerminalBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        terminalbench_generation()
