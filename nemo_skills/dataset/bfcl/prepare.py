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

import subprocess
import os
import glob
import tempfile
import json
import shutil
from copy import deepcopy
from utils import func_doc_language_specific_pre_processing, convert_to_tool, is_multi_turn, load_file
from pathlib import Path
from constants import DATA_FOLDER_PATH, MULTI_TURN_FUNC_DOC_PATH, MULTI_TURN_FUNC_DOC_FILE_MAPPING
import argparse

# Github paths for BFCL
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"


# Define the configuration as a dictionary
DEFAULT_SETTINGS = """
PROMPT_CONFIG = "eval/bfcl/nemotron"
DATASET_GROUP = "tool"
METRICS_TYPE = "bfcl"
EVAL_ARGS = "++eval_type=bfcl"
GENERATION_ARGS = ""
GENERATION_MODULE = "nemo_skills.inference.eval.bfcl"
"""


def process_multi_turn_test_case(instance, repo_root_dir):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    # Mark whether the instance is single-turn or multi-turn. 
    # This is used to determine if the inference should be done in a single turn or multiple turns.
    if not is_multi_turn(instance["id"]):
        instance["single_turn"] = True
        return instance
    else:
        instance["single_turn"] = False

    involved_classes = instance["involved_classes"]
    instance["function"] = []
    for func_collection in involved_classes:
        # func_doc is a list of dict
        func_doc = load_file(
            repo_root_dir / MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        instance["function"].extend(func_doc)

    # Handle Miss Func category; we need to remove the holdout function doc
    if "missed_function" in instance:
        for turn_index, missed_func_names in instance["missed_function"].items():
            instance["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(instance["function"]):
                    if func_doc["name"] == missed_func_name:
                        # Add the missed function doc to the missed_function list
                        instance["missed_function"][turn_index].append(func_doc)
                        # Remove it from the function list
                        instance["function"].pop(i)
                        break

    return instance


def add_tools_to_system_prompt(tools, system_prompt, model_type="nemotron"):
    """Add tools along with their descriptions to the system prompt"""

    if "nemotron" in model_type:
        system_prompt += ('You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[')
        
        for tool in tools:
            # Handle both tool.function and direct tool formats
            tool_def = tool.get('function', tool) if isinstance(tool, dict) and 'function' in tool else tool
            system_prompt += json.dumps(tool_def)
        
        system_prompt += ']</AVAILABLE_TOOLS>\n\n'
        
        # TODO: Move this logic of conditional system prompt to a conditional yaml file (fc vs fc_with_tool)
        # Add tool usage instructions
        system_prompt += (
            'If you decide to call any tool(s), use the following format:\n'
            '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, '
            '{"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n'
            'Response from tool(s) will be returned in this format:\n'
            '<TOOL_RESPONSE>[{"response": "tool_response1"}, {"response": "tool_response2"}]</TOOL_RESPONSE>\n\n'
            'Based on the results returned by the tool(s), you can call additional tools if needed, '
            'correct tool calls if any errors are found, or just respond with the answer to the user.'
        )
        return system_prompt
    else:
         # TODO: Add support for Qwen and maybe other model types
        raise ValueError(f"Model type {model_type} not supported")


def process_file(repo_root_dir, input_file, output_file, model_type="nemotron"):
    """Process a single BFCL file.
    Mainly process the tools and functions and add them to the system prompt.
    Also mark whether the instance is single-turn or multi-turn which is used during inference.
    """

    with open(input_file, "r") as f, open(output_file, "w") as f_out:
        for idx, line in enumerate(f):
            instance = json.loads(line)
            test_category = instance["id"].rsplit("_", 1)[0]
            if idx == 0:                # else:
                print(test_category)            
            
            # Process the data so that the data is amenable to prompt filling
            instruction = ""
            messages = instance["question"]
            for turn in messages:
                for message in turn:
                    if message["role"] == "system":
                        instruction += message["content"]
                    elif message["role"] == "user":
                        # Copying the message to another key to avoid problems with the prompt filling
                        message["query"] = message["content"]
                    elif message["role"] == "assistant":
                        message["response"] = message["content"]

            instance["instruction"] = instruction

            # Convert class-based method calls to function calls
            instance = process_multi_turn_test_case(instance, repo_root_dir)
            
            # Convert function calls to tools format and add them to the system prompt
            if "function" in instance:
                # Add the tools to the system prompt
                instance["function"] = func_doc_language_specific_pre_processing(instance["function"], test_category)
                instance["tools"] = convert_to_tool(instance["function"])
                # Add tools to the system prompt
                instance["instruction"] = add_tools_to_system_prompt(
                    instance["tools"], instance["instruction"], model_type=model_type)
                
            f_out.write(json.dumps(instance) + "\n")


def download_and_process_bfcl_data(repo_url, subfolder_path, output_dir, file_prefix="BFCL_v3", model_type="nemotron"):
    """
    Download JSON files from the BFCL GitHub repo via cloning
    
    Args:
        repo_url: GitHub repository URL
        subfolder_path: Path to the data subfolder in case of BFCL
        output_dir: Directory to save the processed JSONL files
        file_prefix: Only process files starting with this prefix
        model_type: Type of model to process the data for which influences things like system prompt format
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository with minimal depth
            print(f"Cloning repository {repo_url} to {temp_dir}")
            subprocess.run([
                "git", "clone", "--depth=1", repo_url, temp_dir
            ], check=True, capture_output=True)
            
            # Find the target folder
            target_folder = Path(temp_dir) / subfolder_path
            
            if not os.path.exists(target_folder):
                print(f"Folder {subfolder_path} not found in repository")
                raise FileNotFoundError(f"Folder {subfolder_path} not found in {repo_url} cloned to {temp_dir}. The structure of BFCL has changed!")
            
            # Find JSON files matching criteria
            json_pattern = os.path.join(target_folder, f"{file_prefix}*.json")
            json_files = glob.glob(json_pattern)
            
            print(f"Found {len(json_files)} JSON files matching pattern")
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 

            processed_files = 0
            for input_file in json_files:
                filename = os.path.basename(input_file)
                split_dirname = os.path.join(output_dir, filename.lstrip("BFCL_v3_").replace(".json", ""))
                if not os.path.exists(split_dirname):
                    os.makedirs(split_dirname)

                with open(os.path.join(split_dirname, "__init__.py"), "w") as f:
                    f.write(DEFAULT_SETTINGS)

                output_file = os.path.join(split_dirname, "test.jsonl")
                process_file(temp_dir, input_file, output_file, model_type=model_type)

                # Copy the original json file to the split directory
                shutil.copy(input_file, os.path.join(split_dirname, filename))
                processed_files += 1
            
            print(f"Successfully processed {processed_files} JSON files to {output_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            print("Make sure git is installed and the repository URL is correct")


def main(args):
    download_and_process_bfcl_data(
        REPO_URL, DATA_FOLDER_PATH, 
        output_dir=os.path.join(os.path.dirname(__file__)),
        model_type=args.model_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="nemotron", choices=[
        "nemotron", "qwen", "llama-nemotron"])
    args = parser.parse_args()
    main(args)