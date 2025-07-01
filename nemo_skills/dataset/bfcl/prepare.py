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
from utils import func_doc_language_specific_pre_processing, convert_to_tool


# Github paths for BFCL
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
SUBFOLDER_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data"

# Define the configuration as a dictionary
DEFAULT_SETTINGS = """
PROMPT_CONFIG = "eval/bfcl/nemotron"
DATASET_GROUP = "tool"
METRICS_TYPE = "bfcl"
EVAL_ARGS = "++eval_type=bfcl"
GENERATION_ARGS = ""
GENERATION_MODULE = "nemo_skills.inference.eval.bfcl"
"""


def add_tools_to_system_prompt(tools, system_prompt):
    system_prompt += ('You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[')
    
    for tool in tools:
        # Handle both tool.function and direct tool formats
        tool_def = tool.get('function', tool) if isinstance(tool, dict) and 'function' in tool else tool
        system_prompt += json.dumps(tool_def)
    
    system_prompt += ']</AVAILABLE_TOOLS>\n\n'
    
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


def process_file(input_file, output_file):
    all_single = True
    count_complex = 0
    with open(input_file, "r") as f, open(output_file, "w") as f_out:
        for idx, line in enumerate(f):
            instance = json.loads(line)
            test_category = instance["id"].rsplit("_", 1)[0]
            if idx == 0:
                if not "live_" in test_category: #or "multi" in test_category:
                    break
                # else:
                print(test_category)            

            # In this processing, we identify the system content and the user content
            messages = deepcopy(instance["question"])
            system_content = ""
            if len(messages) == 1:
                # Just a single user command
                if len(messages[0]) == 1:
                    assert messages[0][0]["role"] == "user"
                    instance["problem"] = messages[0][0]["content"]

                elif len(messages[0]) == 2:
                    # First is system, second is user for all except one instance
                    if messages[0][0]["role"] == "system":
                        system_content = messages[0][0]["content"]
                        instance["problem"] = messages[0][1]["content"]
                    elif messages[0][0]["role"] == "user":
                        instance["problem"] = messages[0][0]["content"] + " " + messages[0][1]["content"]
                    else:
                        raise ValueError(f"Unknown format of the question: {json.dumps(instance, indent=4)}")

                else:
                    all_single = False
                    count_complex += 1
                    break

                    print("***" * 100)
                    print(json.dumps(instance["question"], indent=4))
                    print("***" * 100)
                    print()
                    print(len(instance["question"]), len(instance["question"][0]))
                    print()
                    # continue
            else:
                # if count_complex == 0:
                #     print("***" * 100)
                #     print(json.dumps(instance["question"], indent=4))
                #     print(len(instance["question"]), len(instance["question"][0]))
                #     print("***" * 100)
                #     print()
                all_single = False
                count_complex += 1
                break
                # continue


            instance["system_content"] = system_content
                
            if "function" in instance:
                # Add the tools to the system prompt
                instance["function"] = func_doc_language_specific_pre_processing(instance["function"], test_category)
                instance["tools"] = convert_to_tool(instance["function"])
                # Add tools to the system prompt
                instance["system_content"] = add_tools_to_system_prompt(
                    instance["tools"], system_content)
                
            f_out.write(json.dumps(instance) + "\n")

    if not all_single:
        print(f"Warning: {input_file} has multiple turns")
    print(f"Count of complex: {count_complex}")


def download_and_process_bfcl_data(repo_url, subfolder_path, output_dir, file_prefix="BFCL_v3"):
    """
    Download JSON files from the BFCL GitHub repo via cloning
    
    Args:
        repo_url: GitHub repository URL
        subfolder_path: Path to the data subfolder in case of BFCL
        output_dir: Directory to save the processed JSONL files
        file_prefix: Only process files starting with this prefix
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository with minimal depth
            print(f"Cloning repository {repo_url} to {temp_dir}")
            subprocess.run([
                "git", "clone", "--depth=1", repo_url, temp_dir
            ], check=True, capture_output=True)
            
            # Find the target folder
            target_folder = os.path.join(temp_dir, subfolder_path)
            
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
                process_file(input_file, output_file)

                # Copy the original json file to the split directory
                shutil.copy(input_file, os.path.join(split_dirname, filename))
                processed_files += 1
            
            print(f"Successfully processed {processed_files} JSON files to {output_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            print("Make sure git is installed and the repository URL is correct")
        # except Exception as e:
        #     print(f"Error: {e}")


if __name__ == "__main__":
    download_and_process_bfcl_data(
        REPO_URL, SUBFOLDER_PATH, 
        output_dir=os.path.join(os.path.dirname(__file__))
    )