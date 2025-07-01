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

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


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
    tool_call_regex) :
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
