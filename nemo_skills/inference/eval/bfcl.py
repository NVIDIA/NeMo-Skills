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
import json
from pathlib import Path
import re   
import shutil
from dataclasses import asdict, field


from nemo_skills.inference.eval.bfcl_utils import extract_tool_response
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.eval.scicode import SciCodeGenerationConfig, SciCodeGenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging
from nemo_skills.prompt.utils import get_prompt
# StatefulMultiTurnPrompt

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLGenerationConfig(GenerateSolutionsConfig):
    """BFCL benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "eval/bfcl/nemotron"
    prompt_template: str = "llama3-instruct"

    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"
    remove_thinking: bool = True

    tool_call_start_token = "<TOOLCALL>"
    tool_call_end_token = "</TOOLCALL>"


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
    
    def preprocess_data(self, data):
        """Preprocess the single-turn instances to create data point level variables."""
        for data_point in data:
            # We process the case where the instance is single-turn
            if data_point["single_turn"]:
                assert len(data_point["question"]) == 1, "Single-turn instances should have exactly one question"

                # Single-turn instance has just one query
                data_point["query"] = ""
                for turn in data_point["question"][0]:
                    if turn["role"] == "user":
                        data_point["query"] += turn["query"]
        
        return data


    def generate_single_data_point_multi_turn(self, data_point):
        """Generate for a single data point with multiple turns."""

        initial_config: dict = data_point["initial_config"]
        involved_classes: list = data_point["involved_classes"]
        test_entry_id: str = data_point["id"]
        test_category: str = data_point["id"].rsplit("_", 1)[0]

        all_model_response: list[list] = []  # The model response that will be used for later evaluation
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed

        inference_data: dict = {}

        # Initialize the stateful prompt 
        prompt = get_prompt(self.cfg.prompt_config, self.cfg.prompt_template, is_stateful=True)

        all_multi_turn_messages: list[list[dict]] = data_point["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]
            if turn_idx == 0:
                if data_point.get("instruction", ""):
                    prompt.add_message({
                        "role": "system", 
                        "content": data_point["instruction"], 
                        "instruction": data_point["instruction"]
                    })

            for message in current_turn_message:
                prompt.add_message(message)                
                
            print(prompt.get_current_prompt())
            # break

            current_turn_response = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )

                input_dict = {
                    "prompts": [prompt.get_current_prompt()],
                    "stop_phrases": prompt.stop_phrases,
                    **asdict(self.cfg.inference),
                    **self.extra_generate_params,
                }

                output = self.llm.generate(**input_dict)[0]

                print(output)

                # Try parsing the model response
                model_response_data = self._process_tool_response(extract_tool_response(
                    output["generation"], 
                    self.cfg.tool_call_start_token, 
                    self.cfg.tool_call_regex
                ))

                print(model_response_data)

                # model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                prompt.add_message(
                    {"role": "assistant", "content": model_response_data, "response": model_response_data}
                )

                current_turn_response.append(model_response_data)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(model_responses)

                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )

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

        metadata = {}
        return all_model_response, metadata


    def llm_generate(self, data_points, data, is_async=True):
        """Depending on whether the instances are single turn or multi-turn, we use different methods to generate."""

        if data_points[0]["single_turn"]:
            return super().llm_generate(data_points, data, is_async=is_async)

        else:
            futures = []
            with ThreadPoolExecutor(max_workers=len(data_points)) as executor:
                for data_point in data_points:
                    future = executor.submit(self.generate_single_data_point_multi_turn, data_point)
                    futures.append(future)

            return futures

    def get_llm_generations(self, requests_in_progress, generations):
        for dp_idx, future in requests_in_progress.items():
            if future.done():
                generations[dp_idx] = future.result()
            else:
                generations[dp_idx] = {'generation': None}

        return requests_in_progress, generations

    def postprocess(self):
        # Extract the tool response from the generation
        print("Postprocessing {}".format(self.cfg.output_file))
        temp_file = Path(self.cfg.output_file).with_suffix(".tmp")

        with open(self.cfg.output_file, "rt", encoding="utf-8") as fin, open(temp_file, "wt", encoding="utf-8") as fout:
            for idx, line in enumerate(fin):
                instance = json.loads(line)
                extracted_tool_response = extract_tool_response(
                    instance["generation"], self.cfg.tool_call_start_token, self.cfg.tool_call_regex)
                instance["result"] = self._process_tool_response(extracted_tool_response)

                fout.write(json.dumps(instance) + "\n")

        shutil.move(temp_file, self.cfg.output_file)

    def _process_tool_response(self, extracted_tool_response):
        """Process the tool response to get the result."""
        return [
            {func_call.function.name: func_call.function.arguments}
            for func_call in extracted_tool_response["tool_calls"]
        ]

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
