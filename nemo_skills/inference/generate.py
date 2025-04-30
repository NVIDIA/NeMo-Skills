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
from nemo_skills.inference.server.code_execution_model import get_code_execution_model, get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import chunk_data, get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.0
    random_seed: int = 0
    tokens_to_generate: int = 2048
    repetition_penalty: float = 1.0
    top_logprobs: int | None = None


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
    # if True, model will be prompted to continue "generation" without closing assistant tag
    continue_prefix_generation: bool = False

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
    # Controls how many code executions are allowed in prompt (useful for models that support dynamically setting this)
    # if total_code_executions placeholder is not in the prompt, this parameter has no effect
    # Can be int, (min,max) tuple, or None
    # If (min,max) tuple, will be randomly sampled from random.randint(min_val, max_val) for each sample in a batch
    # useful to generate data with variable number of total_code_executions_in_prompt
    total_code_executions_in_prompt: Any = None
    # When True, total_code_executions_in_prompt override model defaults
    override_max_code_executions: bool = False

    # extra stop phrases for llms
    extra_stop_phrases: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._post_init_validate_data()
        self._post_init_validate_server()

    def _post_init_validate_data(self):
        if self.input_file is not None:
            if self.dataset is not None or self.split is not None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided, but not both")
        else:
            if self.dataset is None or self.split is None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided")
            self.input_file = Path(__file__).parents[1] / "dataset" / self.dataset / f"{self.split}.jsonl"

        if self.dataset is None and self.prompt_config is None:
            raise ValueError("If `dataset` is not provided, `prompt_config` is required")

        if isinstance(self.total_code_executions_in_prompt, ListConfig):
            self.total_code_executions_in_prompt = list(self.total_code_executions_in_prompt)

        if (self.total_code_executions_in_prompt is not None and 
                not isinstance(self.total_code_executions_in_prompt, (int, list, tuple))):
            raise ValueError(
                "`total_code_executions_in_prompt` must be either int, list, tuple, or None, "
                f"got {type(self.total_code_executions_in_prompt)}"
            )

    def _post_init_validate_server(self):
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

        self.use_async_loop = (
            self.cfg.use_async_loop and self.cfg.server["server_type"] != "nemo" and self.cfg.multi_turn_key is None
        )
        if self.use_async_loop:
            LOG.warning(
                "Async loop is maintaining %d concurrent "
                "requests throughout execution -- batch_size parameter is ignored!\n"
                "Use max_concurrent_requests to control the number of concurrent requests.",
                self.cfg.max_concurrent_requests,
            )

    def setup_llm(self):
        if self.cfg.prompt_template is None and self.cfg.server["server_type"] != "openai":
            with open_dict(self.cfg.server):
                self.cfg.server["server_type"] = "openai"
                self.cfg.server["model"] = "model"
            if self.cfg.code_execution:
                raise ValueError("Code execution is not supported for OpenAI server")

        if self.cfg.code_execution:
            sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
            llm = get_code_execution_model(**self.cfg.server, sandbox=sandbox)
        else:
            llm = get_model(**self.cfg.server)

        return llm

    def setup_prompt(self):
        if self.cfg.prompt_config is None:
            dataset_module = importlib.import_module(f"nemo_skills.dataset.{self.cfg.dataset}")
            self.cfg.prompt_config = dataset_module.PROMPT_CONFIG

        prompt = get_prompt(self.cfg.prompt_config, self.cfg.prompt_template, examples_type=self.cfg.examples_type)
        LOG.info("Prompt used: %s", prompt)
        return prompt

    def log_example_prompt(self, data):
        data_point = deepcopy(data[0])
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

    def skip_completed_samples_sync(self, data):
        if not self.cfg.skip_filled:
            return data

        starting_idx = 0
        if self.cfg.num_chunks:
            chunk_index = self.cfg.output_file.rfind("_chunk")
            base_output_file = self.cfg.output_file[:chunk_index] + ".jsonl"
            if Path(base_output_file).exists():
                LOG.warning(f"File `{base_output_file}` exists, skipping generation")
                return []
        try:
            with open(self.cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{self.cfg.output_file}` not found, starting from scratch")

        if starting_idx > len(data):
            raise ValueError(
                "Number of completed samples is greater than the number of samples "
                "in the dataset (or requested max_samples). Some mistake in configuration?"
            )
        return data[starting_idx:]

    def skip_completed_samples_async(self, data):
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
            dp[self.cfg.async_position_key] = idx
            remaining_data.append(dp)

        return remaining_data

    # TODO: data will not include any samples skipped after restart
    def fill_prompt(self, data_point, data):
        """Passing in full data in case it's needed to fill the prompt in subclasses."""
        total_code_executions_in_prompt = self.cfg.total_code_executions_in_prompt
        if total_code_executions_in_prompt is not None:
            if isinstance(total_code_executions_in_prompt, (list, tuple)):
                min_val, max_val = total_code_executions_in_prompt
                total_code_executions_in_prompt = random.randint(min_val, max_val)
            data_point['total_code_executions'] = total_code_executions_in_prompt
        data_point = deepcopy(data_point)
        return self.prompt.fill(
            data_point,
            multi_turn_key=self.cfg.multi_turn_key,
            prefix_generation_to_response=self.cfg.prefix_generation_to_response,
            continue_prefix_generation=self.cfg.continue_prefix_generation,
        )

    def llm_generate(self, data_points, data, is_async=False):
        generation_params = {
            "prompts": [self.fill_prompt(dp, data) for dp in data_points],
            "stop_phrases": combine_stop_phrases(self.prompt.stop_phrases, self.extra_stop_phrases),
            **asdict(self.cfg.inference),
            **self.extra_generate_params,
        }
        
        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                max_code_executions_values = [
                    dp.get('total_code_executions') for dp in data_points
                ]
                generation_params['max_code_executions'] = max_code_executions_values

        generate_method = self.llm.generate_async if is_async else self.llm.generate
        return generate_method(**generation_params)

    def llm_generate_multi_turn(self, data_points, data):
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
                turn_data_points[dp_index][self.cfg.multi_turn_key] = data_points[dp_index][self.cfg.multi_turn_key][
                    :cur_turn
                ]
                for turn_idx in range(cur_turn - 1):
                    turn_data_points[dp_index][self.cfg.multi_turn_key][turn_idx]['assistant'] = outputs[dp_index][
                        "generation"
                    ][turn_idx]
            # getting a new set of generations
            turn_outputs = self.llm_generate([turn_data_points[dp_index] for dp_index in dp_indices], data)
            # adding assistant answers to the generations
            for pos_index, dp_index in enumerate(dp_indices):
                outputs[dp_index]["generation"].append(turn_outputs[pos_index]["generation"])

            # removing any indices that got through all turns
            dp_indices = []
            for dp_index, (output, dp) in enumerate(zip(outputs, data_points)):
                if len(output["generation"]) < len(dp[self.cfg.multi_turn_key]):
                    dp_indices.append(dp_index)
            cur_turn += 1
        return outputs

    def dump_outputs(self, outputs, data_points, fout):
        for output, original_data_point in zip(outputs, data_points):
            # to make it easier to follow up with evaluation and limit accidental errors, we are adding
            # all of the ground-truth data to the output file alongside the generated solutions
            output[self.cfg.generation_key] = output.pop("generation")
            for key in output:
                original_data_point.pop(key, None)
            output.update(original_data_point)
            fout.write(json.dumps(output) + "\n")

    def sync_loop(self, data):
        with open(self.cfg.output_file, "at", encoding="utf-8", buffering=1) as fout:
            data_points_batch = []
            for idx, data_point in tqdm(enumerate(data), total=len(data), desc="Remaining generations"):
                data_points_batch.append(data_point)
                if len(data_points_batch) == self.cfg.batch_size or idx == len(data) - 1:
                    if self.cfg.multi_turn_key is None:
                        outputs = self.llm_generate(data_points_batch, data)
                    else:
                        outputs = self.llm_generate_multi_turn(data_points_batch, data)
                    self.dump_outputs(outputs, data_points_batch, fout)
                    data_points_batch = []

    def async_loop(self, data):
        pbar = tqdm(total=len(data), desc="Remaining generations")

        last_submitted_idx = 0
        requests_in_progress = {}  # generation_id -> original data_point
        with open(self.cfg.output_file + "-async", "at", encoding="utf-8", buffering=1) as fout:
            while last_submitted_idx < len(data) or len(requests_in_progress) > 0:
                num_to_submit = self.cfg.max_concurrent_requests - len(requests_in_progress)
                if last_submitted_idx < len(data) and num_to_submit > 0:
                    generation_ids = self.llm_generate(
                        data[last_submitted_idx : last_submitted_idx + num_to_submit], data, is_async=True
                    )
                    for idx, gen_id in enumerate(generation_ids):
                        requests_in_progress[gen_id] = data[last_submitted_idx + idx]

                    last_submitted_idx += num_to_submit

                generations = self.llm.get_generations(list(requests_in_progress.keys()))

                outputs_to_dump = []
                data_points_to_dump = []
                for (gen_id, original_dp), gen_dict in zip(requests_in_progress.copy().items(), generations):
                    if gen_dict['generation'] is None:  # not done yet
                        continue
                    # remove the completed task from in_progress
                    requests_in_progress.pop(gen_id)

                    outputs_to_dump.append(gen_dict)
                    data_points_to_dump.append(original_dp)

                    pbar.update(1)

                self.dump_outputs(outputs_to_dump, data_points_to_dump, fout)
                time.sleep(1)  # Prevent excessive API overload

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

        if self.use_async_loop:
            data = self.skip_completed_samples_async(data)
        else:
            data = self.skip_completed_samples_sync(data)

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

        if self.use_async_loop:
            self.async_loop(data)
        else:
            self.sync_loop(data)

        self.postprocess()


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
