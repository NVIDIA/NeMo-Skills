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

import importlib
import json
import logging
from collections import deque
import sys
import time
from copy import deepcopy
from dataclasses import asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import hydra
from omegaconf import OmegaConf, ListConfig, open_dict
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.server.code_execution_model import get_code_execution_model, get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import chunk_data, get_fields_docstring, get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 2048
    repetition_penalty: float = 1.0


@nested_dataclass(kw_only=True)
class GenerateSolutionsConfig:
    """LLM generation parameters."""

    output_file: str  # Where to save the generations
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str | None = None  # we will fetch it from dataset dir if not provided
    prefix_generation_to_response: bool = False  # whether to include "generation" as prefix to the response

    examples_type: str | None = None  # to be able to customize few-shot examples
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    dataset: str | None = None
    split: str | None = None  # Generally one of train/test, but can be anything since it's used as part of a file name
    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset

    batch_size: int = 128
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    max_concurrent_requests: int = 1024  # Maximum number of concurrent requests to the server for the async loop
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    generation_key: str = "generation"
    # if specified, we will have a loop over that key in the data file and
    # treat each element as a new turn of conversation
    # E.g. if multi_turn_key="turns" and a line in your data file has
    # turns: ['Hey how are you?', 'And where do you live?']
    # the generations will also be a list with the first entry corresponding to prompt
    # with the first question, second entry to both first question, first answer and second question
    # and so on
    multi_turn_key: str | None = None

    # set to False if you want to use synchronous loop instead of async. Async loop means we will send all
    # data to engine at the same time (batch size is ignored) and then write the output as soon as it's ready
    # to `output_file`-async (and put it back in order after all generations are done)
    use_async_loop: bool = True
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # set to True if code execution needs to be supported
    code_execution: bool = False

    # extra stop phrases for llms
    extra_stop_phrases: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.input_file is not None:
            if self.dataset is not None or self.split is not None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided, but not both")
        else:
            if self.dataset is None or self.split is None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided")
            self.input_file = Path(__file__).parents[1] / "dataset" / self.dataset / f"{self.split}.jsonl"

        if self.dataset is None and self.prompt_config is None:
            raise ValueError("If `dataset` is not provided, `prompt_config` is required")

        if self.server["server_type"] == "trtllm" and self.prompt_template is None:
            # TODO: fix that
            raise ValueError("Prompt template is required for trtllm servers")

        if self.server["server_type"] == "nemo" and self.prompt_template is None:
            LOG.warning(
                "NeMo implementation of openai chat completions api doesn't support batching and thus is very slow. "
                "Until this is fixed, we highly recommend that you provide prompt template explicitly."
            )

        if self.server["server_type"] == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


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
    def __init__(self, cfg: GenerateSolutionsConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: GenerateSolutionsConfig object with the configuration parameters or subclass.
        """
        self.cfg = cfg
        self.data = None
        self.llm = None
        self.prompt = None

        if not isinstance(self.cfg, GenerateSolutionsConfig):
            raise ValueError(f"Expected GenerateSolutionsConfig or a subclass, got {type(self.cfg)}")

        # Extra parameters for generation
        self.extra_generate_params = {}
        self.extra_stop_phrases = None

        # Synchronous decoding arguments
        self.sync_starting_idx = 0

        # Asynchronous decoding arguments
        self.async_original_positions = None
        self.async_in_progress = None

        # Early exit flag
        self.early_exit = False # Flag to signal early exit from generation

    def load_data(self, cfg: GenerateSolutionsConfig):
        """Template method to load data.

        Note:
        Assigns the loaded data to `self.data`.
        """
        data = []
        with open(cfg.input_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))

        # chunk the dataset if required
        if cfg.num_chunks is not None and cfg.chunk_id is not None:
            data, cfg.output_file = chunk_data(data, cfg.output_file, cfg.chunk_id, cfg.num_chunks)
            LOG.info(
                f"Chunking the data into {cfg.num_chunks} chunks and processing chunk {cfg.chunk_id}.\n"
                f"Number of samples in the chunk: {len(data)}"
            )
        self.data = data

    def sync_data_skip(self, cfg: GenerateSolutionsConfig):
        """Template method to determine data skipping based on output file.

        Logic to skip already filled data based on `cfg.skip_filled` and `cfg.output_file`.
        Also calculates max_samples and early_exit flags.

        Note:
        Updates `self.data` and `self.sync_starting_idx`.
        If self.early_exit is set to True, the template processor must read the value and exit early.
        """
        starting_idx = 0
        if cfg.skip_filled:
            try:
                with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                    starting_idx = len(fin.readlines())
            except FileNotFoundError:
                LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")
        self.sync_starting_idx = starting_idx

        self.data = self.data[self.sync_starting_idx:]

        if 0 <= cfg.max_samples <= starting_idx:
            cfg.max_samples = 0
        elif starting_idx < cfg.max_samples:
            cfg.max_samples -= starting_idx

        data_len = len(self.data) if self.data is not None else 0  # in case data is not yet loaded, use 0
        if cfg.max_samples < 0 or cfg.max_samples > data_len: # use data_len here
            cfg.max_samples = data_len

        if len(self.data) == 0:  # we might not have any examples if skip_filled=True
            self.early_exit = True

        self.data = self.data[: cfg.max_samples]

        if starting_idx >= data_len and cfg.skip_filled:  # check against data_len
            self.early_exit = True
            LOG.info(f"Output file `{cfg.output_file}` is already filled and skip_filled is True. Exiting.")
            return

        self.sync_starting_idx = starting_idx # update instance variable

    def get_llm_and_sandbox(self, cfg: GenerateSolutionsConfig):
        """Template method to get LLM and sandbox.

        Note:
        Assigns the loaded LLM to `self.llm`.
        """
        if cfg.prompt_template is None and cfg.server["server_type"] != "openai":
            with open_dict(cfg.server):
                cfg.server["server_type"] = "openai"
                cfg.server["model"] = "model"
            if cfg.code_execution:
                raise ValueError("Code execution is not supported for OpenAI server")

        if cfg.code_execution:
            sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
            llm = get_code_execution_model(**cfg.server, sandbox=sandbox)
        else:
            llm = get_model(**cfg.server)

        self.llm = llm

    def get_prompt_and_example(self, cfg: GenerateSolutionsConfig):
        """Template method to get prompt and display example.

        Also logs one example from the data with the prompt config + template.

        Note:
        Assigns the loaded prompt to `self.prompt`.
        """
        if cfg.prompt_config is None:
            dataset_module = importlib.import_module(f"nemo_skills.dataset.{cfg.dataset}")
            cfg.prompt_config = dataset_module.PROMPT_CONFIG

        prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
        LOG.info("Prompt used: %s", prompt)

        if cfg.multi_turn_key is None:
            LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", self.data[0], prompt.fill(self.data[0]))
        else:
            first_sample = deepcopy(self.data[0])
            first_sample[cfg.multi_turn_key] = first_sample[cfg.multi_turn_key][:1]
            LOG.info(
                "Example prompt (first turn only):\nData dictionary: %s\nPrompt: %s",
                first_sample,
                prompt.fill(first_sample, multi_turn_key=cfg.multi_turn_key),
            )
        self.prompt = prompt

    def get_extra_parameters(self, cfg: GenerateSolutionsConfig):
        """Template method to get extra parameters for generation.

        Note:
        Assigns the loaded extra parameters to `self.extra_generate_params`.
        """
        if cfg.code_execution:
            self.extra_generate_params = self.prompt.get_code_execution_args()
        else:
            self.extra_generate_params = {}
        self.extra_stop_phrases = OmegaConf.to_container(cfg.extra_stop_phrases, resolve=True)

    def sync_llm_generate_single_turn_hook(self, cfg, data_points):
        """Template method for synchronous LLM generation in sync loop.

        Calculate synchronous generations for single turn data points.
        """
        return self.llm.generate(
                prompts=[
                    self.prompt.fill(dp, prefix_generation_to_response=cfg.prefix_generation_to_response)
                    for dp in data_points
                ],
                stop_phrases=combine_stop_phrases(self.prompt.stop_phrases, self.extra_stop_phrases),
                **asdict(cfg.inference),
                **self.extra_generate_params,
            )

    def sync_llm_generate_multi_turn_hook(self, cfg, data_points):
        """Template method for synchronous LLM generation in sync loop.

        Calculate synchronous generations for multi turn data points.
        """
        # TODO: this will not be efficient if different elements have different number of turns
        # (effective batch size gets smaller). Need to rewrite it to ensure batch size is filled
        # no matter the turns. Also even the below implementation can probably be simplified
        turn_data_points = deepcopy(data_points)
        dp_indices = list(range(len(turn_data_points)))
        cur_turn = 1
        outputs = [{"generation": []} for _ in range(len(data_points))]
        while dp_indices:
            # updating the turns to only have data up-to the current turn
            # and adding any generated assistant messages
            for dp_index in dp_indices:
                turn_data_points[dp_index][cfg.multi_turn_key] = data_points[dp_index][cfg.multi_turn_key][
                                                                 :cur_turn
                                                                 ]
                for turn_idx in range(cur_turn - 1):
                    turn_data_points[dp_index][cfg.multi_turn_key][turn_idx]['assistant'] = outputs[
                        dp_index
                    ]["generation"][turn_idx]
            # getting a new set of generations
            turn_outputs = self.llm.generate(
                prompts=[
                    self.prompt.fill(
                        turn_data_points[dp_index],
                        multi_turn_key=cfg.multi_turn_key,
                        prefix_generation_to_response=cfg.prefix_generation_to_response,
                    )
                    for dp_index in dp_indices
                ],
                stop_phrases=combine_stop_phrases(self.prompt.stop_phrases, self.extra_stop_phrases),
                **asdict(cfg.inference),
                **self.extra_generate_params,
            )
            # adding assistant answers to the generations
            for pos_index, dp_index in enumerate(dp_indices):
                outputs[dp_index]["generation"].append(turn_outputs[pos_index]["generation"])

            # removing any indices that got through all turns
            dp_indices = []
            for dp_index, (output, dp) in enumerate(zip(outputs, data_points)):
                if len(output["generation"]) < len(dp[cfg.multi_turn_key]):
                    dp_indices.append(dp_index)
            cur_turn += 1
        return outputs

    def sync_llm_generate_hook(self, cfg, data_points):
        """Template method for synchronous LLM generation in sync loop.

        Calculate synchronous generations for data points.
        Swaps between single turn and multi turn generation based on `cfg.multi_turn_key`.
        """
        if cfg.multi_turn_key is None:
            # single turn generation
            outputs = self.sync_llm_generate_single_turn_hook(cfg, data_points)

        else:
            # multi-turn generation
            outputs = self.sync_llm_generate_multi_turn_hook(cfg, data_points)
        return outputs

    def sync_output_processing_hook(self, cfg, outputs, data_points, fout):
        """Template method for processing and writing output in sync loop.

        Writes the synchronized output data to the output file.
        Note:
        The output file is not closed so that data appending can be done in the same file.
        """
        for output, original_data_point in zip(outputs, data_points):
            # to make it easier to follow up with evaluation and limit accidental errors, we are adding
            # all of the ground-truth data to the output file alongside the generated solutions
            output[cfg.generation_key] = output.pop("generation")
            for key in output:
                original_data_point.pop(key, None)
            output.update(original_data_point)
            fout.write(json.dumps(output) + "\n")

    def sync_loop(self, cfg: GenerateSolutionsConfig):
        """Outer execution loop for synchronous generation.

        Calls a series of hook methods to perform synchronized decoding and output processing.

        # Order of steps:
        1) sync_llm_generate_hook()
          2) sync_llm_generate_single_turn_hook()
             OR
          3) sync_llm_generate_multi_turn_hook()
        4) sync_output_processing_hook()
        """
        if not self.data:
            return

        with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
            data_points_batch = []
            for idx, data_point in tqdm(enumerate(self.data), initial=self.sync_starting_idx, total=len(self.data) + self.sync_starting_idx):
                data_points_batch.append(data_point)
                if len(data_points_batch) == cfg.batch_size or idx == len(self.data) - 1:
                    outputs = self.sync_llm_generate_hook(cfg, data_points_batch)
                    self.sync_output_processing_hook(cfg, outputs, data_points_batch, fout)
                    data_points_batch = []

    def async_data_skip_positions_hook(self, cfg: GenerateSolutionsConfig) -> List[int]:
        """Template method to determine data skipping based on output file.

        Logic to skip already filled data based on `cfg.skip_filled` and `cfg.output_file`.
        Also calculates max_samples and early_exit flags.

        Returns:
            List of original sample positions to process.
        """
        if cfg.max_samples > 0:
            self.data = self.data[: cfg.max_samples]

        # Compute original position ids for samples
        original_positions = [idx for idx in range(len(self.data))]

        if cfg.skip_filled:
            try:
                filled_positions = set()
                with open(cfg.output_file + '-async', "rt", encoding="utf-8") as fin:
                    for line in fin:
                        filled_positions.add(int(json.loads(line)[cfg.async_position_key]))
                self.data = [dp for idx, dp in enumerate(self.data) if idx not in filled_positions]
                original_positions = [idx for idx in original_positions if idx not in filled_positions]
            except FileNotFoundError:
                LOG.warning(f"File `{cfg.output_file}-async` not found, starting from scratch")

        if Path(cfg.output_file).exists():
            if not cfg.skip_filled:
                Path(cfg.output_file).unlink()
            else:
                self.early_exit = True
                return []

        return original_positions

    def async_get_request_queue(self, cfg: GenerateSolutionsConfig) -> deque:
        """Template method to get request queue for async generation.

        Returns:
            Queue of unsubmitted task indices.
        """
        return deque(range(len(self.data)))

    def async_get_batch_prompts_from_queue(self, cfg: GenerateSolutionsConfig, request_queue: deque) -> Tuple[List[str], List[int]]:
        """Template method to get batch prompts from request queue for async generation.

        Returns:
            Tuple of batch prompts and batch indices.
        """
        # Dynamic sending requests to maintain cfg.max_concurrent_requests running requests
        num_to_submit = min(cfg.max_concurrent_requests - len(self.async_in_progress), len(request_queue))
        batch_indices = [request_queue.popleft() for _ in range(num_to_submit)]
        batch_prompts = [
            self.prompt.fill(self.data[idx], prefix_generation_to_response=cfg.prefix_generation_to_response)
            for idx in batch_indices
        ]

        return batch_prompts, batch_indices

    def async_llm_generate_from_batch_prompts(self, cfg: GenerateSolutionsConfig, batch_prompts: List[str], batch_indices: List[int]) -> Dict[int, str]:
        """Template method to generate LLM output asynchronously for batch prompts.

        Note:
        Updates `self.async_in_progress`.

        Returns:
            Dictionary of generation ids for the async batch.
        """

        if len(batch_prompts) > 0:
            generation_ids = self.llm.generate_async(
                prompts=batch_prompts,
                stop_phrases=combine_stop_phrases(self.prompt.stop_phrases, self.extra_stop_phrases),
                **asdict(cfg.inference),
                **self.extra_generate_params,
            )
        else:
            generation_ids = {}

        # Map the generated ids to the original positions
        for gen_ids_idx, original_pos in enumerate(batch_indices):
            self.async_in_progress[original_pos] = generation_ids[gen_ids_idx]

        return generation_ids

    def async_llm_get_generations(self, cfg: GenerateSolutionsConfig, generation_ids: Dict[int, str]) -> List[Dict[str, str]]:
        """Template method to get LLM generations asynchronously for generation ids.

        Returns:
            List of generations for the async batch.
        """
        # Create a snapshot of in_progress to avoid modifying the dictionary while iterating over it
        snapshot_in_progress = self.async_in_progress.copy()
        generations = self.llm.get_generations(list(snapshot_in_progress.values()))
        return generations

    def async_write_generations(self, cfg: GenerateSolutionsConfig, generations: List[Dict[str, str]], pbar, fout) -> None:
        """Template method to write LLM generations asynchronously.

        Note:
        Updates `self.async_in_progress`.
        """
        # Create a snapshot of in_progress to avoid modifying the dictionary while iterating over it
        snapshot_in_progress = self.async_in_progress.copy()
        for (idx, gen_id), gen_dict in zip(snapshot_in_progress.items(), generations):
            if gen_dict['generation'] is not None:
                # remove the completed task from in_progress
                del self.async_in_progress[idx]

                # Prepare the result for writing
                gen_dict[cfg.generation_key] = gen_dict.pop("generation")
                for key in gen_dict:
                    self.data[idx].pop(key, None)
                gen_dict.update(self.data[idx])

                # Insert the async position to preserve the original order
                gen_dict[cfg.async_position_key] = self.async_original_positions[idx]

                # Write the result immediately to minimize memory usage
                fout.write(json.dumps(gen_dict) + "\n")

                # Update progress bar
                pbar.update(1)

    def async_postprocess_outputs(self, cfg: GenerateSolutionsConfig) -> None:
        """Template method to post-process the outputs of async generation.

        Note:
        Restores the order of generations and resaves the output file without position ids.
        """
        # After we are done, need to restore the order and resave without position ids
        with open(cfg.output_file + '-async', "rt", encoding="utf-8") as fin:
            generations = [json.loads(line) for line in fin]

        ordered_generations = [None] * len(generations)
        for gen_dict in generations:
            async_pos = gen_dict.pop(cfg.async_position_key)
            ordered_generations[async_pos] = gen_dict

        with open(cfg.output_file, "wt", encoding="utf-8") as fout:
            for gen_dict in ordered_generations:
                fout.write(json.dumps(gen_dict) + "\n")

        Path(cfg.output_file + '-async').unlink()

    def async_process_request_queue(self, cfg: GenerateSolutionsConfig, request_queue: deque):
        """Outer execution loop for asynchronous generation.

        Calls a series of hook methods to perform asynchronous decoding and output processing.

        # Order of steps:
        1) async_get_batch_prompts_from_queue()
        2) async_llm_generate_from_batch_prompts()
        3) async_llm_get_generations()
        4) async_write_generations()
        5) async_postprocess_outputs
        """
        with open(cfg.output_file + "-async", "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
            pbar = tqdm(total=len(self.data), desc="Processing requests")

            while self.async_in_progress or request_queue:  # Continue until all tasks are complete
                # Get the batch of prompts to send
                batch_prompts, batch_indices = self.async_get_batch_prompts_from_queue(cfg, request_queue)

                # Get generation ids for the batch
                generation_ids = self.async_llm_generate_from_batch_prompts(cfg, batch_prompts)

                # Get the generations for the batch
                generations = self.async_llm_get_generations(cfg, generation_ids)

                # Process the generations
                self.async_write_generations(cfg, generations, pbar, fout)


                # Prevent excessive API overload
                time.sleep(1)

            pbar.close()

        # Post-process the outputs
        self.async_postprocess_outputs(cfg)


    def async_loop(self, cfg: GenerateSolutionsConfig):
        """Execution processor for asynchronous generation.

        Calls a series of hook methods to perform asynchronous decoding and output processing.

        # Order of steps:
        1) async_data_skip_positions_hook()
        2) async_get_request_queue()
        3) async_process_request_queue()
        """
        self.async_original_positions = self.async_data_skip_positions_hook(cfg)

        if not self.data or self.early_exit:
            LOG.info("No data to process, exiting.")
            return

        LOG.warning(
            f"Async loop is maintaining {cfg.max_concurrent_requests} concurrent requests throughout execution -- batch_size parameter is ignored!."
        )
        LOG.warning("Users can set '++max_concurrent_requests' to control the number of concurrent requests.")

        request_queue = self.async_get_request_queue(cfg) # Queue of unsubmitted task indices
        self.async_in_progress = {}  # Track ongoing requests {index: generation_id}

        self.async_process_request_queue(cfg, request_queue)

    def run_generation_loop(self, cfg: GenerateSolutionsConfig):
        """Template method to run the generation loop (sync or async)."""
        if cfg.use_async_loop is False or cfg.server["server_type"] == "nemo" or cfg.multi_turn_key is not None:
            self.sync_loop(cfg)
        else:
            self.async_loop(cfg)

    def generate_output(self, cfg: GenerateSolutionsConfig):
        """Template method to orchestrate the generation process."""
        Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

        self.load_data(cfg)

        if not cfg.use_async_loop:
            self.sync_data_skip(cfg)

        if self.early_exit: # Check for early exit flag
            LOG.info("Early exit requested, stopping generation.")
            return

        self.get_llm_and_sandbox(cfg)
        self.get_prompt_and_example(cfg)
        self.get_extra_parameters(cfg)

        if cfg.dry_run:
            return

        self.run_generation_loop(cfg)

    @classmethod
    def get_generation_module(cls) -> str:
        """
        Returns the path to the script module that performs the generation task.
        Override this method to customize the generation module.

        Returns:
            str: Path to the generation module.
        """
        return "nemo_skills.inference.generate"

    @classmethod
    def get_generation_default_args(cls) -> str:
        """
        Returns the default arguments for the generation task.
        Override this method to customize the default arguments.

        Returns:
            Dict: Default arguments for the generation task.
        """
        return ""


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_generation_config')
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = GenerationTask(cfg)
    task.generate_output(cfg)


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
