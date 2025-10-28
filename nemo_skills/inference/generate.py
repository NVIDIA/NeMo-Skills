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

import asyncio
import json
import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import hydra
from omegaconf import ListConfig, OmegaConf, open_dict
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.model import (
    OnlineGenSelectConfig,
    get_code_execution_model,
    get_model,
    get_online_genselect_model,
    server_params,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import (
    chunk_data,
    get_help_message,
    get_logger_name,
    nested_dataclass,
    remove_thinking,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = -1
    top_p: float = 0.95
    min_p: float = 0.0
    random_seed: int = 0
    tokens_to_generate: int = 2048
    repetition_penalty: float = 1.0
    top_logprobs: int | None = None

    extra_body: dict = field(default_factory=dict)  # Any other extra params passed with extra_body argument


@nested_dataclass(kw_only=True)
class GenerateSolutionsConfig:
    """LLM generation parameters."""

    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations
    prompt_config: str | None = None  # How to format the data into prompts
    prompt_template: str | None = None  # not required for OpenAI server
    # to specify the format of the prompt, "ns" for NeMo-Skills format or "openai" for OpenAI chat format
    prompt_format: str = "ns"
    prompt_suffix: str = ""  # suffix to add to the prompt, e.g. " /no_think"
    system_message: str | None = None  # can override the default system message in the config
    code_tags: str | None = None  # required when using code execution
    examples_type: str | None = None  # to be able to customize few-shot examples

    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prefix_generation_to_response: bool = False  # whether to include "generation" as prefix to the response
    # if True, model will be prompted to continue "generation" without closing assistant tag
    continue_prefix_generation: bool = False

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

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
    # if specified, we will have a loop over that key in the data file and
    # treat each element as a new turn of conversation
    # E.g. if multi_turn_key="turns" and a line in your data file has
    # turns: ['Hey how are you?', 'And where do you live?']
    # the generations will also be a list with the first entry corresponding to prompt
    # with the first question, second entry to both first question, first answer and second question
    # and so on
    multi_turn_key: str | None = None

    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # set to True if code execution needs to be supported
    code_execution: bool = False
    # Controls how many code executions are allowed in prompt (useful for models that support dynamically setting this)
    # if total_code_executions placeholder is not in the prompt, this parameter has no effect
    # Can be int, (min,max) tuple, or None
    # If (min,max) tuple, will be randomly sampled from random.randint(min_val, max_val) for each sample in a batch
    # useful to generate data with variable number of total_code_executions_in_prompt
    total_code_executions_in_prompt: Any = None
    # When True, total_code_executions_in_prompt override model defaults
    override_max_code_executions: bool = False

    # set to True if online genselect is used
    online_genselect: bool = False
    # genselect config
    online_genselect_config: OnlineGenSelectConfig = field(default_factory=OnlineGenSelectConfig)

    # extra stop phrases for llms
    extra_stop_phrases: list[str] = field(default_factory=list)

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    remove_thinking: bool = False
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"

    def __post_init__(self):
        self._post_init_validate_data()
        self._post_init_validate_server()
        self._post_init_validate_params()

    def _post_init_validate_data(self):
        if isinstance(self.total_code_executions_in_prompt, ListConfig):
            self.total_code_executions_in_prompt = list(self.total_code_executions_in_prompt)

        if self.total_code_executions_in_prompt is not None and not isinstance(
            self.total_code_executions_in_prompt, (int, list, tuple)
        ):
            raise ValueError(
                "`total_code_executions_in_prompt` must be either int, list, tuple, or None, "
                f"got {type(self.total_code_executions_in_prompt)}"
            )

    def _post_init_validate_server(self):
        if self.server["server_type"] in ["nemo", "megatron"] and self.prompt_template is None:
            LOG.warning(
                "NeMo/Megatron implementation of openai chat completions api "
                "doesn't support batching and thus is very slow. "
                "Until this is fixed, we highly recommend that you provide prompt template explicitly."
            )

        if self.server["server_type"] in ["openai", "azureopenai"] and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"
            assert self.prompt_template is None, "prompt_template is not supported for prompt_format == 'openai'"
            assert self.system_message is None, "system_message is not supported for prompt_format == 'openai'"
        else:
            assert self.prompt_config is not None, "prompt_config is required when prompt_format == 'ns'"
        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return []


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


def combine_stop_phrases(prompt_phrases, extra_phrases):
    if prompt_phrases is None and extra_phrases is None:
        return None
    if prompt_phrases is None:
        return extra_phrases
    if extra_phrases is None:
        return prompt_phrases

    if isinstance(extra_phrases, ListConfig):
        extra_phrases = OmegaConf.to_object(extra_phrases)

    return prompt_phrases + extra_phrases


class GenerationTask:
    @classmethod
    def get_generation_default_args(cls) -> str:
        """
        Returns the default arguments for the generation task.
        Override this method to customize the default arguments.

        Returns:
            Dict: Default arguments for the generation task.
        """
        return ""

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """
        Returns the function to get the server command for the generation task.
        Override this method to customize the server command function.

        Returns:
            callable: Function that returns the server command.
        """
        from nemo_skills.pipeline.utils import get_server_command

        return get_server_command

    def __init__(self, cfg: GenerateSolutionsConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: GenerateSolutionsConfig object with the configuration parameters or subclass.
        """
        self.cfg = cfg

        self.llm = self.setup_llm()
        self.prompt = self.setup_prompt()

        if self.cfg.code_execution:
            self.extra_generate_params = self.prompt.get_code_execution_args()
        else:
            self.extra_generate_params = {}

        self.extra_stop_phrases = OmegaConf.to_container(self.cfg.extra_stop_phrases, resolve=True)

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )

        # Initialize semaphore for controlling concurrent requests
        if self.cfg.online_genselect:
            # Each request will generate multiple solutions, so we need to divide the semaphore by the parallel requests
            self.semaphore = asyncio.Semaphore(
                self.cfg.max_concurrent_requests // self.cfg.online_genselect_config.max_concurrent_requests
            )
        else:
            self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

    def setup_llm(self):
        if self.cfg.code_execution:
            sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
            llm = get_code_execution_model(**self.cfg.server, sandbox=sandbox)
        elif self.cfg.online_genselect:
            # Use the same prompt template for genselect as the one used for generation
            self.cfg.online_genselect_config.prompt_template = self.cfg.prompt_template
            self.cfg.online_genselect_config.thinking_begin = self.cfg.thinking_begin
            self.cfg.online_genselect_config.thinking_end = self.cfg.thinking_end
            llm = get_online_genselect_model(
                **self.cfg.server, online_genselect_config=self.cfg.online_genselect_config
            )
        else:
            llm = get_model(**self.cfg.server)

        return llm

    def setup_prompt(self):
        if self.cfg.prompt_format == "openai":
            return None

        prompt = get_prompt(
            self.cfg.prompt_config, self.cfg.prompt_template, self.cfg.code_tags, examples_type=self.cfg.examples_type
        )
        if self.cfg.system_message is not None:
            prompt.config.system = self.cfg.system_message
        LOG.info("Prompt used: %s", prompt)
        return prompt

    def log_example_prompt(self, data):
        data_point = deepcopy(data[0])

        if self.cfg.prompt_format == "openai":
            # print the prompt in openai format
            LOG.info("Example prompt in OpenAI format: \nData dictionary: %s", data_point)
            return

        if self.cfg.multi_turn_key is None:
            LOG.info(
                "Example prompt:\nData dictionary: %s\nPrompt: %s", data_point, self.fill_prompt(data_point, data)
            )
        else:
            data_point[self.cfg.multi_turn_key] = data_point[self.cfg.multi_turn_key][:1]
            LOG.info(
                "Example prompt (first turn only):\nData dictionary: %s\nPrompt: %s",
                data_point,
                self.fill_prompt(data_point, data),
            )

    def load_data(self):
        data = []
        with open(self.cfg.input_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))

        # chunk the dataset if required
        if self.cfg.num_chunks is not None and self.cfg.chunk_id is not None:
            data, self.cfg.output_file = chunk_data(data, self.cfg.output_file, self.cfg.chunk_id, self.cfg.num_chunks)
            LOG.info(
                f"Chunking the data into {self.cfg.num_chunks} chunks and processing chunk {self.cfg.chunk_id}.\n"
                f"Number of samples in the chunk: {len(data)}"
            )

        if self.cfg.max_samples > 0:
            data = data[: self.cfg.max_samples]

        return data

    def preprocess_data(self, data):
        """A placeholder for any data preprocessing that needs to be done before generation."""
        return data

    def postprocess(self):
        """A placeholder for any postprocessing that needs to be done after generation.

        Data is already saved to self.cfg.output_file, so it can be read and re-saved from there.
        """
        pass

    def skip_completed_samples(self, data):
        # if non-async file exists and we are asked to skip filled, then there is no more data to process
        if self.cfg.skip_filled and Path(self.cfg.output_file).exists():
            return []

        filled_positions = set()
        if self.cfg.skip_filled:
            if self.cfg.num_chunks:
                chunk_index = self.cfg.output_file.rfind("_chunk")
                base_output_file = self.cfg.output_file[:chunk_index] + ".jsonl"
                if Path(base_output_file).exists():
                    LOG.warning(f"File `{base_output_file}` exists, skipping generation")
                    return []
            try:
                with open(self.cfg.output_file + '-async', "rt", encoding="utf-8") as fin:
                    for line in fin:
                        filled_positions.add(int(json.loads(line)[self.cfg.async_position_key]))
            except FileNotFoundError:
                LOG.warning(f"File `{self.cfg.output_file}-async` not found, starting from scratch")

        remaining_data = []
        for idx, dp in enumerate(data):
            if idx in filled_positions:
                continue
            if self.cfg.prompt_format == "openai" and isinstance(dp, list):
                # openai format allows for a list to be top-level key, if that's the case, wrapping it in a messages key
                dp = {"messages": dp}
            dp[self.cfg.async_position_key] = idx
            remaining_data.append(dp)

        return remaining_data

    # TODO: data will not include any samples skipped after restart
    def fill_prompt(self, data_point, data):
        """Passing in full data in case it's needed to fill the prompt in subclasses."""
        if self.cfg.prompt_format == "openai":
            if self.cfg.prompt_suffix:
                data_point["messages"][-1]["content"] += self.cfg.prompt_suffix
            return data_point["messages"]

        total_code_executions_in_prompt = self.cfg.total_code_executions_in_prompt
        if total_code_executions_in_prompt is not None:
            if isinstance(total_code_executions_in_prompt, (list, tuple)):
                min_val, max_val = total_code_executions_in_prompt
                total_code_executions_in_prompt = random.randint(min_val, max_val)
            data_point['total_code_executions'] = total_code_executions_in_prompt
        data_point = deepcopy(data_point)
        filled_prompt = self.prompt.fill(
            data_point,
            multi_turn_key=self.cfg.multi_turn_key,
            prefix_generation_to_response=self.cfg.prefix_generation_to_response,
            continue_prefix_generation=self.cfg.continue_prefix_generation,
        )
        if self.cfg.prompt_suffix:
            if isinstance(filled_prompt, list):
                filled_prompt[-1]['content'] += self.cfg.prompt_suffix
            else:
                filled_prompt += self.cfg.prompt_suffix
        return filled_prompt

    def dump_outputs(self, outputs, data_points, fout):
        for output, original_data_point in zip(outputs, data_points):
            # to make it easier to follow up with evaluation and limit accidental errors, we are adding
            # all of the ground-truth data to the output file alongside the generated solutions
            output[self.cfg.generation_key] = output.pop("generation")

            # calculating total generation time
            if self.cfg.add_generation_stats:
                output['generation_end_time'] = time.time()
                # TODO: start time is saved in data_point, not output, need to fix that
                output['generation_time'] = (
                    output['generation_end_time'] - original_data_point['generation_start_time']
                )
            else:
                # generation_start_time was overriden, so restoring it from end and total
                # TODO: this is a bit hacky, need a rewrite
                if 'generation_end_time' in original_data_point and 'generation_time' in original_data_point:
                    output['generation_start_time'] = (
                        original_data_point['generation_end_time'] - original_data_point['generation_time']
                    )
                else:
                    output.pop('generation_start_time', None)
                output.pop('num_generated_tokens', None)

            for key in output:
                original_data_point.pop(key, None)
            output.update(original_data_point)
            if self.cfg.remove_thinking:
                remove_thinking(output, self.cfg.generation_key, self.cfg.thinking_begin, self.cfg.thinking_end)
            fout.write(json.dumps(output) + "\n")

    def prefill_generation(self, data_point) -> dict | None:
        """Prefill generation in case LLM is not required."""
        # Override this method to customize the prefilling behavior.
        return None

    async def process_single_datapoint(self, data_point, all_data):
        generation_params = {
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": combine_stop_phrases(
                self.prompt.stop_phrases if self.prompt is not None else None, self.extra_stop_phrases
            ),
            **asdict(self.cfg.inference),
            **self.extra_generate_params,
        }

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                max_code_executions_values = [data_point['total_code_executions']]
                generation_params['max_code_executions'] = max_code_executions_values

        return await self.llm.generate_async(**generation_params)

    async def _process_single_datapoint_with_semaphore(self, data_point, all_data, fout, pbar):
        """Process a single data point with semaphore control."""
        async with self.semaphore:
            # registering current time to calculate total generation time
            data_point['generation_start_time'] = time.time()

            # Generate output for this single data point
            output = await self.process_single_datapoint(data_point, all_data)

            # Thread-safe output writing
            async with self.output_lock:
                self.dump_outputs([output], [data_point], fout)
                pbar.update(1)

    async def async_loop(self, data):
        """Async loop to generate generations using asyncio."""

        # Initialize output lock for thread-safe writing
        if self.output_lock is None:
            self.output_lock = asyncio.Lock()

        # We first segregate the data into prefilled and non-prefilled data points
        prefilled_data_points, prefilled_outputs = [], []
        remaining_data_points = []

        for data_point in data:
            prefill_output = self.prefill_generation(data_point)
            if prefill_output is not None:
                prefilled_outputs.append(prefill_output)
                prefilled_data_points.append(data_point)
            else:
                remaining_data_points.append(data_point)

        LOG.info("Testing if async loop is working...")
        pbar = tqdm(total=len(remaining_data_points), desc="Remaining generations")
        LOG.info("Testing if async loop is working... done")

        with open(self.cfg.output_file + "-async", "at", encoding="utf-8", buffering=1) as fout:
            # Dump prefilled data first
            if len(prefilled_data_points) > 0:
                async with self.output_lock:
                    self.dump_outputs(prefilled_outputs, prefilled_data_points, fout)

            # Create tasks for all remaining data points
            tasks = []
            for data_point in remaining_data_points:
                task = asyncio.create_task(self._process_single_datapoint_with_semaphore(data_point, data, fout, pbar))
                tasks.append(task)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

            pbar.close()

        self.restore_async_order()

    def restore_async_order(self):
        # After we are done, need to restore the order and resave without position ids
        with open(self.cfg.output_file + '-async', "rt", encoding="utf-8") as fin:
            generations = [json.loads(line) for line in fin]

        ordered_generations = [None] * len(generations)
        for gen_dict in generations:
            async_pos = gen_dict.pop(self.cfg.async_position_key)
            ordered_generations[async_pos] = gen_dict

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for gen_dict in ordered_generations:
                fout.write(json.dumps(gen_dict) + "\n")

        Path(self.cfg.output_file + '-async').unlink()

    def generate(self):
        Path(self.cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

        data = self.load_data()

        data = self.skip_completed_samples(data)

        if len(data) == 0:
            LOG.info("No data to process, exiting.")
            return

        data = self.preprocess_data(data)

        self.log_example_prompt(data)

        if self.cfg.dry_run:
            LOG.info("Exiting without running generation as dry_run flag is set.")
            return

        if not self.cfg.skip_filled:
            for output_path in [Path(self.cfg.output_file), Path(self.cfg.output_file + "-async")]:
                if output_path.exists():
                    output_path.unlink()

        asyncio.run(self.async_loop(data))

        print("Async loop completed")
        self.postprocess()

        # Clean up the LLM client to prevent hanging
        if hasattr(self.llm, 'close'):
            self.llm.close()


GENERATION_TASK_CLASS = GenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_generation_config')
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = GenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
