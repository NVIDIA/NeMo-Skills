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
import logging
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import List
from openai import BadRequestError

import hydra

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.model import (
    get_model,
    server_params,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    remove_thinking,
    setup_logging,
)

from .lean4_utils import *
from .generate import GenerateSolutionsConfig, GenerationTask

LOG = logging.getLogger(get_logger_name(__file__))

reasoning_effort_list = ["low", "medium", "high"]


@nested_dataclass(kw_only=True)
class AutoformalizeConfig(GenerateSolutionsConfig):
    """LLM generation parameters."""

    # Lean 4 specific parameters
    refine_parsing_error_prompt_config: str | None = (
        None  # prompt for refining the code
    )
    refine_code_error_prompt_config: str | None = None  # prompt for refining the code
    refine_consistent_error_prompt_config: str | None = (
        None  # prompt for refining the code
    )
    refinement: bool = False  # whether to refine the code
    refinement_max_turns: int = 8  # maximum number of turns for refinement
    judge_enabled: bool = False  # whether to judge the code
    backtranslation_prompt_config: str | None = None  # prompt for backtranslation
    judge_prompt_config: str | None = None  # prompt for judging the code
    judge_exact_match: bool = (
        True  # recommend to set to true when using gpt-oss and should set to flase if using deepseek
    )
    adaptive_reasoning: bool = False  # whether to adapt the reasoning effort
    parse_generation: bool = False  # whether to parse the generation


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=AutoformalizeConfig)


class AutoformalizeTask(GenerationTask):

    def __init__(self, cfg: AutoformalizeConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: AutoformalizeConfig object with the configuration parameters or subclass.
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
        if self.cfg.judge_enabled:
            self.setup_judge_prompt()

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

        self.output_lock = None

    def setup_llm(self):
        if self.cfg.code_execution:
            raise ValueError("Code execution is not supported for autoformalization")
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
            self.cfg.refine_parsing_error_prompt_config is not None
        ), "refine_parsing_error_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        assert (
            self.cfg.refine_code_error_prompt_config is not None
        ), "refine_code_error_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        if self.cfg.judge_enabled:
            assert (
                self.cfg.refine_consistent_error_prompt_config is not None
            ), "refine_consistent_error_prompt_config is required when refinement is enabled and judge is enabled. Please set refinement=False to disable refinement."
        self.refine_parsing_error_prompt = get_prompt(
            self.cfg.refine_parsing_error_prompt_config
        )
        self.refine_code_error_prompt = get_prompt(
            self.cfg.refine_code_error_prompt_config
        )
        self.refine_consistent_error_prompt = get_prompt(
            self.cfg.refine_consistent_error_prompt_config
        )

    def setup_judge_prompt(self):
        assert (
            self.cfg.backtranslation_prompt_config is not None
        ), "backtranslation_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        assert (
            self.cfg.judge_prompt_config is not None
        ), "judge_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        self.judge_prompt = get_prompt(self.cfg.judge_prompt_config)
        self.backtranslation_prompt = get_prompt(self.cfg.backtranslation_prompt_config)

    def _extract_code_sync(self, completion: str):
        try:
            code = extract_code(completion)
            if code == "None":
                return None, None
            clean_code = remove_comments(code)
            clean_code = move_imports_to_beginning(clean_code)
            clean_code = refine_by_sorry(clean_code)
            return code, clean_code
        except:
            return None, None

    async def _extract_code(self, completion: str):
        # Offload the blocking work to another thread
        return await asyncio.to_thread(self._extract_code_sync, completion)

    async def _backtranslate_code(self, code: str) -> str:
        prompt = self.backtranslation_prompt.fill({"code": code})
        generation = await self._generate_single_completion(prompt)
        try:
            backtranslation_result = generation["generation"]
        except:
            backtranslation_result = None
        return backtranslation_result

    async def _judge_backtranslation(
        self, backtranslation_result: str, data_point
    ) -> str:
        prompt = self.judge_prompt.fill(
            {
                "backtranslation": backtranslation_result,
                "problem": data_point["problem"],
            }
        )
        generation = await self._generate_single_completion(prompt)
        try:
            judge_result = generation["generation"]
        except:
            judge_result = None
        return judge_result

    async def _judge_code(self, code: str | None, data_point) -> dict:
        results_dict = {}
        results_dict["code"] = code
        results_dict["passed_compile"] = False
        results_dict["backtranslation_result"] = None
        results_dict["judge_result"] = None
        results_dict["passed_compile_judge"] = False
        results_dict["feedback"] = None
        if code is None:
            results_dict["parse_error"] = True
            return results_dict
        else:
            results_dict["parse_error"] = False
        code_execution_result = await self.llm.sandbox.execute_lean4_code(
            remove_comments(code)
        )
        results_dict["code_execution_result"] = code_execution_result

        if (
            type(code_execution_result) == dict
            and code_execution_result["process_status"] == "completed"
        ):
            results_dict["passed_compile"] = True
            if self.cfg.judge_enabled:
                backtranslation_result = await self._backtranslate_code(code)
                if backtranslation_result is not None:
                    results_dict["backtranslation_result"] = backtranslation_result
                    judge_result = await self._judge_backtranslation(
                        backtranslation_result, data_point
                    )
                    results_dict["judge_result"] = judge_result
                    if judge_result is not None:
                        if self.cfg.judge_exact_match:
                            if "true" == judge_result.lower().strip():
                                results_dict["passed_compile_judge"] = True
                        else:
                            if "true" in judge_result.lower().strip():
                                results_dict["passed_compile_judge"] = True
                    else:
                        results_dict["passed_compile_judge"] = True
                        results_dict["judge_result"] = (
                            "Backtranslation passed, but judge failed. Skipping..."
                        )
                else:
                    results_dict["backtranslation_result"] = (
                        "Backtranslation failed. Skipping..."
                    )
                    results_dict["passed_compile_judge"] = True
            else:
                results_dict["passed_compile_judge"] = True
        elif type(code_execution_result) == str:
            results_dict["code_execution_result"] = {
                "process_status": "failed",
                "stdout": "Timeout error, pleasee check for heavy computation, dead loop, etc.",
            }
        return results_dict

    def _construct_refine_prompt(self, results_dict):
        if results_dict["parse_error"]:
            # parse error
            prompt = self.refine_parsing_error_prompt.fill({})
        elif results_dict["passed_compile"]:
            # consistent error
            prompt = self.refine_consistent_error_prompt.fill(
                {"reason": results_dict["judge_result"]}
            )
        else:
            # code error
            prompt = self.refine_code_error_prompt.fill(
                {"error_message": results_dict["code_execution_result"]["stdout"]}
            )
        return prompt

    async def _generate_single_completion(self, prompt: List[str]):

        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)
        generation_params = {
            "prompt": prompt,
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            **inference_params,
            **self.extra_generate_params,
        }
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

    async def _signle_data_point_generate(self, data_point, data):

        results_dict = {}
        prompt_turn_list = self.fill_prompt(data_point, data)
        code_list = []
        unrefined_code_list = []
        results_dict_list = []
        assert type(prompt_turn_list) == list, "prompt_turn_list should be a list"
        results_dict["passed_compile_judge"] = False

        try:
            for turn_idx in range(self.cfg.refinement_max_turns):
                generation = await self._generate_single_completion(prompt_turn_list)
                prompt_turn_list += [
                    {"role": "assistant", "content": generation["generation"]}
                ]
                unrefined_code, code = await self._extract_code(
                    generation["generation"]
                )
                unrefined_code_list.append(unrefined_code)
                code_list.append(code)
                results_dict = await self._judge_code(code, data_point)
                if "reasoning_content" in generation:
                    results_dict["reasoning_content_generation"] = generation[
                        "reasoning_content"
                    ]
                results_dict_list.append(results_dict)
                if results_dict["passed_compile_judge"]:
                    break
                else:
                    if (
                        self.cfg.refinement
                        and turn_idx < self.cfg.refinement_max_turns - 1
                    ):
                        prompt = self._construct_refine_prompt(results_dict)
                        results_dict["feedback"] = prompt
                        prompt_turn_list += prompt
                    else:
                        break
        except BadRequestError as e:
            print("BadRequestError: context window too long")
        return {
            "code_list": code_list,
            "unrefined_code_list": unrefined_code_list,
            "results_dict_list": results_dict_list,
            "prompt_turn_list": prompt_turn_list,
            "turn_idx": turn_idx,
            "success": results_dict["passed_compile_judge"],
        }

    async def process_single_datapoint(self, data_point, all_data):
        result = await self._signle_data_point_generate(data_point, all_data)
        result_dict = {"generation": result}
        return result_dict


GENERATION_TASK_CLASS = AutoformalizeTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: AutoformalizeConfig):
    cfg = AutoformalizeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = AutoformalizeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    AutoformalizeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
