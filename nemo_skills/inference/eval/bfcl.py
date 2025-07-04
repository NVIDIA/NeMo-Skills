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

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

import hydra
import re   
from dataclasses import asdict, field

from nemo_skills.inference.eval.bfcl_utils import convert_to_function_call, execute_multi_turn_func_call, is_empty_execute_response, MAXIMUM_STEP_LIMIT, DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

from nemo_skills.dataset.bfcl.utils import func_doc_language_specific_pre_processing, convert_to_tool

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLGenerationConfig(GenerateSolutionsConfig):
    """BFCL benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "generic/default"
    prompt_template: str = "default-base"

    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"
    remove_thinking: bool = True

    tool_call_start_token = "<TOOLCALL>"
    tool_call_end_token = "</TOOLCALL>"

    system_prompt: str = "detailed thinking on\n\n"

    @property
    def tool_call_regex(self):
        """Compiled regex pattern for extracting tool calls."""
        return re.compile(
            r"{}(.*?){}".format(
                re.escape(self.tool_call_start_token), 
                re.escape(self.tool_call_end_token)
            ), 
            re.DOTALL
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_bfcl_generation_config", node=BFCLGenerationConfig)


class BFCLGenerationTask(GenerationTask):
    def __init__(self, cfg: BFCLGenerationConfig):
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
                    "but BFCL evaluation requires it for efficient inference. "
                    "Each request will be processed 1 by 1, which is extremely inefficient and slow! "
                    "We highly recommend switching to a server that supports inflight batching."
                )
        self.use_async_loop = True

    def log_example_prompt(self, data):
        """BFCL is a multi-turn benchmark, so we can't print a single prompt."""
        return


    def _generate_single_assistant_turn(self, inference_state_dict):
        """Generate for a single assistant turn."""
        messages = inference_state_dict["messages"]
        tools = inference_state_dict["tools"]

        if self.cfg.system_prompt:
            messages = [{"role": "system", "content": self.cfg.system_prompt}] + messages


        input_dict = {
            "prompts": [messages],
            "tools": [tools],
            **asdict(self.cfg.inference),
            **self.extra_generate_params,
        }

        output = self.llm.generate(**input_dict)[0]
        if "tool_calls" not in output:
            output["tool_calls"] = []

        return output

    
    def generate_single_data_point_single_turn(self, data_point):
        """Generate for a single data point with a single turn."""
        state_dict = {"messages": data_point["question"][0], "tools": data_point["tools"]}

        model_response = self._generate_single_assistant_turn(state_dict)
        proc_model_response = self._process_model_response(model_response)

        return {"id": data_point["id"], "generation": proc_model_response["generation"], "num_generated_tokens": model_response.get("num_generated_tokens", 0)}


    def generate_single_data_point_multi_turn(self, data_point):
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

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_response = []
            count = 0
            
            if str(turn_idx) in holdout_function:
                data_point["function"].extend(holdout_function[str(turn_idx)])
                # Need to recompile the tools
                functions = func_doc_language_specific_pre_processing(data_point["function"], test_category)
                tools = convert_to_tool(functions)
                state_dict["tools"] = tools

                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]
            
            state_dict["messages"].extend(current_turn_message)

            while True:
                model_response = self._generate_single_assistant_turn(state_dict)
                output_dict["num_generated_tokens"] += model_response.get("num_generated_tokens", 0)
                output_dict["log_dict_list"].append(model_response)
                
                # Add the message to the state dict for chat history
                state_dict["messages"].append(model_response["message"])

                # Proccess the model response text
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
                    model_name="nemotron",  # Can be any placeholder
                    test_entry_id=test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                for execution_result, tool_call_id in zip(
                    execution_results, proc_model_response["tool_call_ids"]
                ):
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
                    print(f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.")
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)

            if force_quit:
                break

        return {"generation": all_model_response, "num_generated_tokens": output_dict["num_generated_tokens"]}
    
    def llm_generate(self, data_points, data, is_async=True):
        """Depending on whether the instances are single turn or multi-turn, we use different methods to generate."""

        if data_points[0]["single_turn"]:
            method = self.generate_single_data_point_single_turn
        else:
            method = self.generate_single_data_point_multi_turn

        futures = []
        with ThreadPoolExecutor(max_workers=len(data_points)) as executor:
            for data_point in data_points:
                future = executor.submit(method, data_point)
                futures.append(future)

        return futures


    def get_llm_generations(self, requests_in_progress, generations):
        for dp_idx, future in requests_in_progress.items():
            if future.done():
                generations[dp_idx] = future.result()
            else:
                generations[dp_idx] = {'generation': None}

        return requests_in_progress, generations
    
    def _process_model_response_text(self, model_response_text):
        """Process the model response text to remove the thinking text if needed."""
        if model_response_text is None:
            return ""
        else:
            if self.cfg.remove_thinking:
                if self.cfg.thinking_end in model_response_text:
                    return model_response_text.split(self.cfg.thinking_end)[-1].lstrip('\n')
            else:
                # Thinking didn't finish
                if self.cfg.remove_thinking:
                    return ""
                else:
                    return model_response_text
    
    def _process_model_response(self, model_response):
        """Process the model response to get the result."""
        try:
            generation = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in model_response["tool_calls"]
            ]
            tool_call_ids = [
                func_call.id for func_call in model_response["tool_calls"]
            ]
        except:
            generation = model_response["generation"]
            tool_call_ids = []

        return {
            "generation": generation,
            "tool_call_ids": tool_call_ids,
            # The original data structure is needed for the chat history
            "message": model_response["message"],
        }


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
