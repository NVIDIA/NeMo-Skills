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

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, field
from functools import partial

import hydra
import openai
from omegaconf import OmegaConf

from nemo_skills.dataset.bfcl_v3.utils import convert_to_tool, func_doc_language_specific_pre_processing
from nemo_skills.inference.eval.bfcl_utils import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
    convert_to_function_call,
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLGenerationConfig(GenerateSolutionsConfig):
    """BFCL benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    remove_thinking: bool = True
    use_client_parsing: bool = True
    model_name: str | None = None

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        from bfcl_eval.constants.model_config import local_inference_model_map

        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"

        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

        if self.use_client_parsing:
            if self.model_name is None:
                raise ValueError("model_name is required when use_client_parsing is True")

            # Add FC by default
            if "-FC" not in self.model_name[-3:]:
                LOG.info(f"Assuming the function calling version of model is being used: {self.model_name}")
                self.model_name += "-FC"

            if self.model_name not in local_inference_model_map:
                # TODO: We can present the user the nearest model name that is supported
                raise ValueError(
                    f"{self.model_name} is not supported by BFCL Eval. "
                    f"Supported models: {list(local_inference_model_map.keys())}"
                )

            LOG.info(f"Using client parsing for {self.model_name}")

            # There are two key functionalities that we need to support on the client side:
            # 1. Parse the response and extract the tool calls
            # 2. Format the prompt

            # 1. Initialize the response parser
            model_handler_class = local_inference_model_map[self.model_name].model_handler
            # Initialize the model handler - Temperature is not used but required by the model handler
            model_handler = model_handler_class(self.model_name, temperature=self.inference.temperature)
            # We only need the response parser from the model handler
            self.response_parser = model_handler._parse_query_response_prompting

            # 2. Initialize the prompt formatter
            # While BFCL model_handler also has the _format_prompt method, we found errors in it's implementation
            # So we use the tokenizer to format the prompt instead which uses the chat template directly
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_handler.model_name_huggingface)
            self.message_formatter = partial(
                self.tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True
            )

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("prompt_config", None),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_bfcl_generation_config", node=BFCLGenerationConfig)


class BFCLGenerationTask(GenerationTask):
    def __init__(self, cfg: BFCLGenerationConfig):
        super().__init__(cfg)

    def log_example_prompt(self, data):
        """BFCL is a multi-turn benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        return None

    async def _generate_single_assistant_turn(self, inference_state_dict):
        """Generate for a single assistant turn."""
        messages = inference_state_dict["messages"]
        tools = inference_state_dict["tools"]

        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages

        # Step 1: Construct the prompt
        if self.cfg.use_client_parsing:
            fmted_prompt = self.cfg.message_formatter(messages, tools=tools)
            input_dict = {
                "prompt": fmted_prompt,
                "include_response": True,
                **asdict(self.cfg.inference),
                **self.extra_generate_params,
            }
        else:
            input_dict = {
                "prompt": messages,
                "tools": tools,
                "include_response": True,
                **asdict(self.cfg.inference),
                **self.extra_generate_params,
            }

        # Step 2: Query the LLM server
        # Enable soft-fail when the models run out of context
        try:
            output = await self.llm.generate_async(**input_dict)
        # TODO: Currently we're assuming an openai interface which is not true for all servers
        except openai.BadRequestError as e:
            error_str = str(e)
            context_error = "is longer than the model's context length" in error_str
            token_error = "Requested token count exceeds model's maximum context length" in error_str

            if context_error or token_error:
                LOG.warning(f"BFCL generation failed due to running out of context. {error_str}")
                return {"message": None, "generation": ""}
            else:
                raise

        # Step 3: Parse the generated output. In case of server side parsing, merely getting the response message
        if self.cfg.use_client_parsing:
            parsed_response = self.cfg.response_parser(output["response"])["model_responses_message_for_chat_history"]

            model_response = {
                "role": "assistant",
                "content": parsed_response["content"],
            }
            if "tool_calls" in parsed_response:
                model_response["tool_calls"] = parsed_response["tool_calls"]

            return {
                # Message is a turn formatted in chat format which gets appended to the chat history
                "message": model_response,
                # Generation is either the text or is empty if there are tool calls
                "generation": parsed_response["content"],
                "tool_calls": parsed_response.get("tool_calls", []),
                "num_generated_tokens": output["num_generated_tokens"],
            }
        else:
            output["message"] = output["response"].choices[0].message
            output["tool_calls"] = []
            if output["message"].tool_calls:
                output["tool_calls"] = output["message"].tool_calls

            return output

    async def generate_single_data_point_single_turn(self, data_point):
        """Generate for a single data point with a single turn."""
        state_dict = {"messages": data_point["question"][0], "tools": data_point["tools"]}

        model_response = await self._generate_single_assistant_turn(state_dict)

        if model_response["message"] is None:
            # Ran out of context
            return {"generation": "", "num_generated_tokens": 0, "error": "_ran_out_of_context_"}
        else:
            proc_model_response = self._process_model_response(model_response)
            return {
                "generation": proc_model_response["generation"],
                "num_generated_tokens": model_response.get("num_generated_tokens", 0),
            }

    async def generate_single_data_point_multi_turn(self, data_point):
        """Generate for a single data point with multiple turns."""

        initial_config: dict = data_point["initial_config"]
        involved_classes: list = data_point["involved_classes"]
        test_entry_id: str = data_point["id"]
        test_category: str = data_point["id"].rsplit("_", 1)[0]
        holdout_function: dict[int, list] = data_point.get("missed_function", {})

        all_model_response: list[list] = []  # The model response that will be used for later evaluation
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed

        all_multi_turn_messages: list[list[dict]] = data_point["question"]
        state_dict = {"messages": [], "tools": data_point["tools"]}
        output_dict = {"result": [], "num_generated_tokens": 0, "log_dict_list": []}
        out_of_context = False

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_response = []
            count = 0

            if str(turn_idx) in holdout_function:
                data_point["function"].extend(holdout_function[str(turn_idx)])
                # Need to recompile the tools
                functions = func_doc_language_specific_pre_processing(data_point["function"], test_category)
                tools = convert_to_tool(functions)
                state_dict["tools"] = tools

                assert len(current_turn_message) == 0, "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            state_dict["messages"].extend(current_turn_message)

            while True:
                model_response = await self._generate_single_assistant_turn(state_dict)
                if model_response["message"] is None:
                    # Ran out of context
                    out_of_context = True
                    LOG.info("Quitting the multi-turn generation due to running out of context.")
                    break

                output_dict["num_generated_tokens"] += model_response.get("num_generated_tokens", 0)
                output_dict["log_dict_list"].append(model_response)

                if self.cfg.remove_thinking:
                    if self.cfg.use_client_parsing:
                        if model_response["message"]["content"] is not None:
                            trimmed_content = self._process_model_response_text(model_response["message"]["content"])
                            model_response["message"]["content"] = trimmed_content
                    else:
                        if model_response["message"].content is not None:
                            model_response["message"].content = self._process_model_response_text(
                                model_response["message"].content
                            )

                # Add the message to the state dict for chat history
                state_dict["messages"].append(model_response["message"])

                # Process the model response text
                proc_model_response = self._process_model_response(model_response)
                # Add the processed model response to the current turn responses
                current_turn_response.append(proc_model_response["generation"])

                # Try decoding the model response
                try:
                    decoded_model_responses = convert_to_function_call(proc_model_response["generation"])
                    if is_empty_execute_response(decoded_model_responses):
                        LOG.info("Empty response from the model. Proceed to next turn.")
                        break

                except Exception as e:
                    LOG.info("Failed to decode the model response. Proceed to next turn.")
                    break

                # Obtain the execution results
                # TODO: Move the execution to sandbox
                execution_results, _ = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    test_entry_id=test_entry_id,
                    long_context=("long_context" in test_category or "composite" in test_category),
                )

                # Add the execution results to the chat history for the next turn
                for execution_result, tool_call_id in zip(execution_results, proc_model_response["tool_call_ids"]):
                    tool_message = {
                        "role": "tool",
                        "content": execution_result,
                        "tool_call_id": tool_call_id,
                    }
                    state_dict["messages"].append(tool_message)

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    LOG.info(f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.")
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)

            if force_quit or out_of_context:
                break

        output_dict["generation"] = all_model_response

        if out_of_context:
            output_dict["error"] = "_ran_out_of_context_"

        return output_dict

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single data point and return the result."""
        if data_point["single_turn"]:
            return await self.generate_single_data_point_single_turn(data_point)
        else:
            return await self.generate_single_data_point_multi_turn(data_point)

    def _process_model_response(self, model_response):
        """Process the model response to get the result."""
        try:
            if self.cfg.use_client_parsing:
                generation = [
                    {func_call["name"]: json.dumps(func_call["arguments"])}
                    for func_call in model_response["tool_calls"]
                ]
                tool_call_ids = [idx for idx in range(len(generation))]
            else:
                generation = [
                    {func_call.function.name: func_call.function.arguments}
                    for func_call in model_response["tool_calls"]
                ]
                tool_call_ids = [func_call.id for func_call in model_response["tool_calls"]]
        except:
            generation = model_response["generation"]
            tool_call_ids = []

        return {
            "generation": generation,
            "tool_call_ids": tool_call_ids,
            # The original data structure is needed for the chat history
            "message": model_response["message"],
        }

    def _process_model_response_text(self, model_response_text):
        if self.cfg.thinking_end in model_response_text:
            return model_response_text.split(self.cfg.thinking_end)[-1].lstrip('\n')
        else:
            # If the thinking didn't finish, we can keep it empty
            return ""


GENERATION_TASK_CLASS = BFCLGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_bfcl_generation_config')
def bfcl_generation(cfg: BFCLGenerationConfig):
    cfg = BFCLGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = BFCLGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    BFCLGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        bfcl_generation()
