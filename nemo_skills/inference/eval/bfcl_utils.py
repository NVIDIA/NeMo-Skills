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
import json
import re
import copy
import inspect
import importlib

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system",
    "MathAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.math_api",
    "MessageAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.message_api",
    "TwitterAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.posting_api",
    "TicketAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.ticket_api",
    "TradingBot": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot",
    "TravelAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.travel_booking",
    "VehicleControlAPI": "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.vehicle_control",
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = [
    "MathAPI",
]

MAXIMUM_STEP_LIMIT = 20


class DotDict(dict):
    """Dictionary that supports dot notation access"""
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            elif isinstance(value, list):
                return [DotDict(item) if isinstance(item, dict) else item for item in value]
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


def convert_to_function_call(function_call_list):
    if type(function_call_list) == dict:
        function_call_list = [function_call_list]
    # function_call_list is of type list[dict[str, str]] or list[dict[str, dict]]
    execution_list = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if type(value) == str:
                value = json.loads(value)
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
            )

    return execution_list


def extract_tool_response(
    model_output: str, 
    tool_call_start_token: str,
    tool_call_end_token: str,
    tool_call_regex, 
) :
    """
    Extract tool calls from model output and format them for BFCL evaluation.
    
    Args:
        model_output: The raw output from the language model
        tool_call_start_token: Token that indicates the start of tool calls
        tool_call_regex: Compiled regex pattern to extract tool calls
        
    Returns:
        Dictionary containing:
        - tools_called: Boolean indicating if tools were called
        - tool_calls: List of formatted tool call dictionaries
        - content: The content portion of the response (before tool calls)
    """
    # Check if tool calls are present in the output
    if tool_call_start_token not in model_output:
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output,
        }
    
    try:
        # Extract tool calls using regex
        regex_matches = tool_call_regex.findall(model_output)
        if not regex_matches:
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": model_output,
            }
            
        str_tool_calls = regex_matches[0].strip()
        
        # Fix JSON formatting if needed
        if not str_tool_calls.startswith("["):
            str_tool_calls = "[" + str_tool_calls
        if not str_tool_calls.endswith("]"):
            str_tool_calls = str_tool_calls + "]"
        
        # Parse the JSON tool calls
        json_tool_calls = json.loads(str_tool_calls)
        tool_calls = []
        
        # Process each tool call
        for tool_call in json_tool_calls:
            try:
                # Convert to the format expected by BFCL
                formatted_tool_call = {
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": (
                            json.dumps(tool_call["arguments"], ensure_ascii=False) 
                            if isinstance(tool_call["arguments"], dict) 
                            else tool_call["arguments"]
                        ),
                    }
                }
                tool_calls.append(DotDict(formatted_tool_call))
            except Exception as parse_error:
                # Skip malformed tool calls but continue processing others
                print(f"Warning: Failed to parse tool call {tool_call}: {parse_error}")
                continue
        
        # Extract content before the tool call
        tool_call_position = model_output.rfind(tool_call_start_token)
        content = model_output[:tool_call_position] if tool_call_position > 0 else None
        

        return {
            "tools_called": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content": content.strip() if content else None,
        }
        
    except json.JSONDecodeError as json_error:
        LOG.error(f"JSON decode error in tool call extraction: {json_error}")
        LOG.error(f"Problematic JSON string: {str_tool_calls}")
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output,
        }
    except Exception as e:
        LOG.error(f"Error in extracting tool call from response.")
        LOG.error(f"Response: {model_output}")
        LOG.error(f"Error: {e}")
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output,
        }


def execute_multi_turn_func_call(
    func_call_list: list[str],  # a list of strings of func calls
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
) -> tuple[list[str], dict]:
    """
    TODO: Add docstring
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # TODO: Handler the model name issue from handler more elegantly
        instance_name = (
            f"sample_model_{test_entry_id}_{class_name.lower()}_instance"
        )
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                # Deep copy the initial configuration to avoid mutation issues
                class_instance._load_scenario(
                    copy.deepcopy(class_initial_config), long_context=long_context
                )
            globals()[instance_name] = class_instance
        # This happens in subsequent turns
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        # Retrieve all method names and map them to the instance
        for method_name, method in inspect.getmembers(
            class_instance, predicate=inspect.ismethod
        ):
            # Skip private methods
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error. 
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if type(func_call_result) == str:
                pass
            elif type(func_call_result) == dict:
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        class_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string
