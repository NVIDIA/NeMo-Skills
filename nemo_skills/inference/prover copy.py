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
from typing import Any, List

import re
import hydra
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.model import (
    OnlineGenSelectConfig,
    get_code_execution_model,
    get_model,
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
from openai import BadRequestError

from .lean4_utils import *
from .generate import GenerateSolutionsConfig, GenerationTask

LOG = logging.getLogger(get_logger_name(__file__))

reasoning_effort_list = [
    "low",
    "medium",
    "high",
]  # This is only used for adaptive reasoning with gpt-oss models


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.6  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.0
    random_seed: int = 0
    tokens_to_generate: int = 40960
    repetition_penalty: float = 1.0
    top_logprobs: int | None = None
    extra_body: dict = field(
        default_factory=dict
    )  # Any other extra params passed with extra_body argument


@nested_dataclass(kw_only=True)
class ProverConfig(GenerateSolutionsConfig):
    """LLM generation parameters."""

    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations
    prompt_config: str | None = None  # How to format the data into prompts

    use_completions_api: bool = False
    # path or name of the tokenizer to use for completions API. By default uses server.model
    tokenizer: str | None = None

    max_tokens: int = 40960  # model max tokens
    n_pass: int = 1  # number of passes to run the prover

    # Lean 4 specific parameters
    refinement: bool = False  # whether to refine the code
    refinement_max_turns: int = 2  # maximum number of turns for refinement
    refinement_prompt_config: str | None = None  # prompt for refining the code
    adaptive_reasoning: bool = False  # whether to adapt the reasoning effort
    parse_generation: bool = False  # whether to parse the generation
    remove_cot: bool = False  # whether to remove the cot from the generation
    delete_wrong_turns: bool = (
        False  # whether to delete the wrong turns from the generation
    )
    chat_template_kwargs: dict = field(default_factory=dict)

    # to specify the format of the prompt, "ns" for NeMo-Skills format or "openai" for OpenAI chat format
    prompt_format: str = "ns"
    prompt_suffix: str = ""  # suffix to add to the prompt, e.g. " /no_think"
    system_message: str | None = (
        None  # can override the default system message in the config
    )
    code_tags: str | None = None  # required when using code execution
    examples_type: str | None = None  # to be able to customize few-shot examples

    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    start_assistant_response_key: str | None = (
        None  # whether to start assistant response with this key
    )

    inference: InferenceConfig = field(
        default_factory=InferenceConfig
    )  # LLM call parameters

    max_samples: int = (
        -1
    )  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = (
        False  # If True, will skip the generations that are already in the output file
    )

    # maximum number of concurrent requests to the server for the async loop
    # if sync loop is used, this is the batch size
    max_concurrent_requests: int = 512
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = (
        None  # if specified, will split the data into chunks and only generate for one chunk
    )
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    # if False, will not add num_generated_tokens and generation_time values.
    # Useful when running judge jobs to keep the original generation statistics
    add_generation_stats: bool = True

    generation_key: str = "generation"
    async_position_key: str = (
        "_async_position"  # key to use for preserving position in async loop in data dict
    )

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
    # stop phrase for llms
    stop_phrase: str | None = None  # if None, will not add any extra stop phrase

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
            self.total_code_executions_in_prompt = list(
                self.total_code_executions_in_prompt
            )

        if self.total_code_executions_in_prompt is not None and not isinstance(
            self.total_code_executions_in_prompt, (int, list, tuple)
        ):
            raise ValueError(
                "`total_code_executions_in_prompt` must be either int, list, tuple, or None, "
                f"got {type(self.total_code_executions_in_prompt)}"
            )

    def _post_init_validate_server(self):
        if self.server["server_type"] == "megatron":
            if self.tokenizer is None:
                raise ValueError(
                    "Megatron server doesn't support chat completions and we can't infer tokenizer from model name. "
                    "Please provide it with an explicit `tokenizer` parameter."
                )
            self.cfg.use_completions_api = True
            LOG.warning(
                "Megatron inference is extremely slow. It's highly recommended to use other server types!"
            )

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(
                f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'"
            )

        if self.prompt_format == "openai":
            assert (
                self.prompt_config is None
            ), "prompt_config is not supported for prompt_format == 'openai'"
        else:
            assert (
                self.prompt_config is not None
            ), "prompt_config is required when prompt_format == 'ns'"
        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

        if self.n_pass > 32:
            raise ValueError("Please consider using ")

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return []


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


class ProverTask:
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
        # chat template kwargs goes either into extra body of inference or as a prompt parameter

        if self.cfg.chat_template_kwargs:
            if not self.cfg.use_completions_api:
                if "chat_template_kwargs" in self.cfg.inference.extra_body:
                    raise ValueError(
                        "chat_template_kwargs is provided in both inference.extra_body and as a separate argument. "
                        "You can only use one of them!"
                    )
                self.cfg.inference.extra_body = dict(self.cfg.inference.extra_body)
                self.cfg.inference.extra_body["chat_template_kwargs"] = dict(
                    self.cfg.chat_template_kwargs
                )
                self.cfg.chat_template_kwargs = None

        self.llm = self.setup_llm()
        self.prompt = self.setup_prompt()
        if self.cfg.refinement:
            self.setup_refine_prompt()

        if self.cfg.code_execution:
            self.extra_generate_params = self.prompt.get_code_execution_args()
        else:
            self.extra_generate_params = {}

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )

        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

        if self.cfg.delete_wrong_turns:
            assert (
                self.cfg.remove_cot
            ), "remove_cot is required when delete_wrong_turns is enabled"

    def setup_llm(self):
        if self.cfg.code_execution:
            raise ValueError("Code execution is not supported for prover")
        sandbox = (
            get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
        )
        server = deepcopy(self.cfg.server)
        server["server_type"] = "autoformalization"
        llm = get_model(**server, sandbox=sandbox)
        return llm

    def setup_prompt(self):
        if self.cfg.prompt_format == "openai":
            return None
        if self.cfg.use_completions_api:
            tokenizer = self.cfg.tokenizer or self.cfg.server["model"]
        else:
            tokenizer = None
        prompt = get_prompt(
            prompt_config=self.cfg.prompt_config,
            tokenizer=tokenizer,
            code_tags=self.cfg.code_tags,
            examples_type=self.cfg.examples_type,
        )
        if self.cfg.system_message is not None:
            prompt.config.system = self.cfg.system_message
        LOG.info("Prompt used: %s", prompt)
        return prompt

    def setup_refine_prompt(self):
        assert (
            self.cfg.refinement_prompt_config is not None
        ), "refinement_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        self.refine_prompt = get_prompt(self.cfg.refinement_prompt_config)

    def setup_judge_prompt(self):
        assert (
            self.cfg.backtranslation_prompt_config is not None
        ), "backtranslation_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        assert (
            self.cfg.judge_prompt_config is not None
        ), "judge_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        self.judge_prompt = get_prompt(self.cfg.judge_prompt_config)
        self.backtranslation_prompt = get_prompt(self.cfg.backtranslation_prompt_config)

    def log_example_prompt(self, data):
        data_point = deepcopy(data[0])
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s",
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
            data, self.cfg.output_file = chunk_data(
                data, self.cfg.output_file, self.cfg.chunk_id, self.cfg.num_chunks
            )
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
                    LOG.warning(
                        f"File `{base_output_file}` exists, skipping generation"
                    )
                    return []
            try:
                with open(
                    self.cfg.output_file + "-async", "rt", encoding="utf-8"
                ) as fin:
                    for line in fin:
                        filled_positions.add(
                            int(json.loads(line)[self.cfg.async_position_key])
                        )
            except FileNotFoundError:
                LOG.warning(
                    f"File `{self.cfg.output_file}-async` not found, starting from scratch"
                )

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
            if self.cfg.system_message:
                if data_point["messages"][0]["role"] != "system":
                    data_point["messages"].insert(
                        0, {"role": "system", "content": self.cfg.system_message}
                    )
                else:
                    data_point["messages"][0]["content"] = self.cfg.system_message
            return data_point["messages"]

        total_code_executions_in_prompt = self.cfg.total_code_executions_in_prompt
        if total_code_executions_in_prompt is not None:
            if isinstance(total_code_executions_in_prompt, (list, tuple)):
                min_val, max_val = total_code_executions_in_prompt
                total_code_executions_in_prompt = random.randint(min_val, max_val)
            data_point["total_code_executions"] = total_code_executions_in_prompt
        data_point = deepcopy(data_point)
        filled_prompt = self.prompt.fill(
            data_point,
            start_assistant_response_key=self.cfg.start_assistant_response_key,
            chat_template_kwargs=self.cfg.chat_template_kwargs,
        )
        if self.cfg.prompt_suffix:
            if isinstance(filled_prompt, list):
                filled_prompt[-1]["content"] += self.cfg.prompt_suffix
            else:
                filled_prompt += self.cfg.prompt_suffix
        return filled_prompt

    # with adaptive reasoning
    async def _generate_single_completion(self, prompt: List[str], **kwargs):
        generation_params = {
            "prompt": prompt,
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            **asdict(self.cfg.inference),
            **self.extra_generate_params,
        }
        for key, value in kwargs.items():
            generation_params[key] = value
        generation = await self.llm.generate_async(**generation_params)
        if self.cfg.adaptive_reasoning:
            assert (
                generation_params["extra_body"].get("reasoning_effort", None)
                is not None
            ), "reasoning_effort is required when adaptive_reasoning is enabled"
            reasoning_effort_index = reasoning_effort_list.index(
                generation_params["extra_body"].get("reasoning_effort", None)
            )
            while len(generation["generation"]) == 0 and reasoning_effort_index > 0:
                print(
                    f"Reasoning effort is too high, reducing to {reasoning_effort_list[reasoning_effort_index-1]}"
                )
                reasoning_effort_index = reasoning_effort_index - 1
                generation_params["extra_body"]["reasoning_effort"] = (
                    reasoning_effort_list[reasoning_effort_index]
                )
                generation = await self.llm.generate_async(**generation_params)
        if self.cfg.parse_generation:
            remove_thinking(
                generation,
                self.cfg.generation_key,
                self.cfg.thinking_begin,
                self.cfg.thinking_end,
            )
        return generation

    # factor out his part so it won't become a bottleneck.
    async def _extract_and_replace_code(self, formal_statement, generation):
        code = extract_code(generation)
        full_code = replace_statement_in_proof(formal_statement, code)
        return code, full_code

    async def _signle_data_point_generate(self, data_point, data):

        formal_statement = (
            (data_point["header"].strip() + "\n")
            + data_point["informal_prefix"].strip()
            + ("\n" + data_point["formal_statement"].strip())
        )
        formal_statement = refine_by_sorry(formal_statement)
        prompt_turn_list = self.prompt.fill({"problem": formal_statement.strip()})

        full_prompt_turn_list = deepcopy(
            prompt_turn_list
        )  # We need to get a full copy of the prompt turn list for the final result in case remove_cot is enabled. This is only used to generate SFT data.
        promt_turn_list_list = (
            []
        )  # We need to store the prompt turn list for each turn for the final result in case delete_wrong_turns is enabled. This is only used to generate SFT data.
        base_prompt_turn_list = deepcopy(prompt_turn_list)

        code_list = []
        results_dict_list = []
        assert type(prompt_turn_list) == list, "prompt_turn_list should be a list"

        success = False
        for turn_idx in range(self.cfg.refinement_max_turns):
            results_dict = {}  # everything will be stored in this dict
            prefix_tokens = self.llm.tokenizer.apply_chat_template(
                prompt_turn_list, tokenize=True
            )
            num_tokens_prefix = len(prefix_tokens)
            prefix = self.llm.tokenizer.apply_chat_template(
                prompt_turn_list, tokenize=False
            )
            # We need to check if the prefix is too long, if it is, we need to break the loop
            if num_tokens_prefix > self.cfg.max_tokens:
                break

            generation = await self._generate_single_completion(
                prefix,
                tokens_to_generate=min(
                    self.cfg.max_tokens - num_tokens_prefix,
                    self.cfg.inference.tokens_to_generate,
                ),
            )

            new_prompt_turn_list = deepcopy(prompt_turn_list)
            new_prompt_turn_list += [
                {"role": "assistant", "content": generation["generation"]}
            ]

            promt_turn_list_list.append(
                new_prompt_turn_list
            )  # This stores the latest turn list after each generation.

            code, full_code = await self._extract_and_replace_code(
                formal_statement, generation["generation"]
            )
            code_list.append(full_code)
            results_dict["code"] = code  # We keep track of the uncleaned code.
            if self.cfg.remove_cot and not (
                code == "None" or "**Error**" in full_code
            ):  # check if successfully parse the code. We do not want to delete the turn if there is a parsing error.
                if self.cfg.delete_wrong_turns:
                    prompt_turn_list = deepcopy(base_prompt_turn_list) + [
                        {
                            "role": "assistant",
                            "content": f"```lean4\n{full_code.strip()}\n```",
                        }
                    ]  # only keep the latest turn
                else:
                    prompt_turn_list += [
                        {
                            "role": "assistant",
                            "content": f"```lean4\n{full_code.strip()}\n```",
                        }
                    ]
                full_prompt_turn_list += [
                    {"role": "assistant", "content": generation["generation"]}
                ]
            else:
                prompt_turn_list += [
                    {"role": "assistant", "content": generation["generation"]}
                ]
                full_prompt_turn_list += [
                    {"role": "assistant", "content": generation["generation"]}
                ]

            if code == "None" or "**Error**" in full_code:
                if code == "None":
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": "Parsing error. Cannot parse the code from output. Please try again and write the code in the format of ```lean4\n<code>\n```",
                    }
                elif "**Error**" in full_code:
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": full_code,
                    }
                results_dict["execution_result"] = execution_result
                results_dict["success"] = False
                feedback = self.refine_prompt.fill(
                    {"error_message": execution_result["stdout"]}
                )
                results_dict["feedback"] = feedback[0]["content"]
            else:
                execution_result = await self.llm.sandbox.execute_lean4_code(
                    full_code, timeout=60.0
                )
                results_dict["execution_result"] = execution_result
                if type(execution_result) == dict:
                    if (
                        execution_result["process_status"] == "completed"
                        and "sorry" not in execution_result["stdout"]
                        and "failed" not in execution_result["stdout"]
                    ):
                        results_dict["success"] = True
                    else:
                        error_list = parse_error(execution_result["stdout"])
                        error_message = get_error_str(
                            full_code, error_list, error_thres=True
                        )
                        feedback = self.refine_prompt.fill(
                            {
                                "error_message": "We use <error></error> to signal the position of the error. \n"
                                + error_message
                            }
                        )
                        results_dict["feedback"] = feedback[0]["content"]
                        results_dict["success"] = False
                elif (
                    type(execution_result) == str
                ):  # This is only used for the case when the code execution timed out.
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": execution_result,
                    }
                    results_dict["success"] = False
                    feedback = self.refine_prompt.fill(
                        {
                            "error_message": "The compilation timed out. There might be a heavy computation in the code or an endless loop."
                        }
                    )
                    results_dict["feedback"] = feedback[0]["content"]
                else:
                    raise ValueError(
                        f"Unknown execution result type: {type(execution_result)}"
                    )

            results_dict_list.append(results_dict)

            if results_dict["success"]:
                # This is the case when the code execution is successful. The theorem is proved.
                break
            else:
                if self.cfg.refinement and turn_idx < self.cfg.refinement_max_turns - 1:
                    prompt_turn_list += feedback
                    full_prompt_turn_list += feedback
                else:
                    # Proving attempt failed.
                    break

        if len(results_dict_list) > 0 and results_dict_list[-1]["success"]:
            success = True

        # Usually only need prompt_turn_list for standard SFT, full_prompt_turn_list for SFT with remove_cot enabled, promt_turn_list_list for SFT with delete_wrong_turns enabled.
        return {
            "code_list": code_list,
            "results_dict_list": results_dict_list,
            "prompt_turn_list": prompt_turn_list,
            "turn_idx": turn_idx,
            "success": success,
            "full_prompt_turn_list": full_prompt_turn_list,
            "promt_turn_list_list": promt_turn_list_list,
        }

    async def pass_at_N(self, data_point, data, N=None):
        if N is None:
            N = self.cfg.n_pass

        new_results_dict = {"success": False}
        # results_dict_list = []
        for i in range(N):
            results_dict = await self._signle_data_point_generate(data_point, data)
            # results_dict_list.append(results_dict)

            if results_dict["success"]:
                new_results_dict["success"] = True
                break

        new_results_dict["results_dict_list"] = results_dict
        new_results_dict["n_pass"] = i + 1

        return new_results_dict

    def dump_outputs(self, outputs, data_points, fout):
        for output, original_data_point in zip(outputs, data_points):
            # to make it easier to follow up with evaluation and limit accidental errors, we are adding
            # all of the ground-truth data to the output file alongside the generated solutions
            output[self.cfg.generation_key] = output.pop("generation")

            # calculating total generation time
            if self.cfg.add_generation_stats:
                output["generation_end_time"] = time.time()
                # TODO: start time is saved in data_point, not output, need to fix that
                output["generation_time"] = (
                    output["generation_end_time"]
                    - original_data_point["generation_start_time"]
                )
            else:
                # generation_start_time was overriden, so restoring it from end and total
                # TODO: this is a bit hacky, need a rewrite
                if (
                    "generation_end_time" in original_data_point
                    and "generation_time" in original_data_point
                ):
                    output["generation_start_time"] = (
                        original_data_point["generation_end_time"]
                        - original_data_point["generation_time"]
                    )
                else:
                    output.pop("generation_start_time", None)
                output.pop("num_generated_tokens", None)

            for key in output:
                original_data_point.pop(key, None)
            output.update(original_data_point)
            if self.cfg.remove_thinking:
                remove_thinking(
                    output,
                    self.cfg.generation_key,
                    self.cfg.thinking_begin,
                    self.cfg.thinking_end,
                )
            fout.write(json.dumps(output) + "\n")

    def prefill_generation(self, data_point) -> dict | None:
        """Prefill generation in case LLM is not required."""
        # Override this method to customize the prefilling behavior.
        return None

    async def process_single_datapoint(self, data_point, all_data):

        result = await self.pass_at_N(data_point, all_data)
        result_dict = {"generation": result}

        return result_dict

    async def _process_single_datapoint_with_semaphore(
        self, data_point, all_data, fout, pbar
    ):
        """Process a single data point with semaphore control."""
        async with self.semaphore:
            # registering current time to calculate total generation time
            data_point["generation_start_time"] = time.time()

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

        pbar = tqdm(total=len(remaining_data_points), desc="Remaining generations")

        with open(
            self.cfg.output_file + "-async", "at", encoding="utf-8", buffering=1
        ) as fout:
            # Dump prefilled data first
            if len(prefilled_data_points) > 0:
                async with self.output_lock:
                    self.dump_outputs(prefilled_outputs, prefilled_data_points, fout)

            # Create tasks for all remaining data points
            tasks = []
            for data_point in remaining_data_points:
                task = asyncio.create_task(
                    self._process_single_datapoint_with_semaphore(
                        data_point, data, fout, pbar
                    )
                )
                tasks.append(task)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

            pbar.close()

        self.restore_async_order()

    def restore_async_order(self):
        # After we are done, need to restore the order and resave without position ids
        with open(self.cfg.output_file + "-async", "rt", encoding="utf-8") as fin:
            generations = [json.loads(line) for line in fin]

        ordered_generations = [None] * len(generations)
        for gen_dict in generations:
            async_pos = gen_dict.pop(self.cfg.async_position_key)
            ordered_generations[async_pos] = gen_dict

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for gen_dict in ordered_generations:
                fout.write(json.dumps(gen_dict) + "\n")

        Path(self.cfg.output_file + "-async").unlink()

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
            for output_path in [
                Path(self.cfg.output_file),
                Path(self.cfg.output_file + "-async"),
            ]:
                if output_path.exists():
                    output_path.unlink()

        asyncio.run(self.async_loop(data))

        self.postprocess()


GENERATION_TASK_CLASS = ProverTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ProverTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
