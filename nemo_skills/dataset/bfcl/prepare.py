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
from utils import func_doc_language_specific_pre_processing


# Github paths for BFCL
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
SUBFOLDER_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data"

# Define the configuration as a dictionary
DEFAULT_SETTINGS = """
PROMPT_CONFIG = "generic/bfcl"
DATASET_GROUP = "tool"
METRICS_TYPE = "bfcl"
EVAL_ARGS = "++eval_type=bfcl"
GENERATION_ARGS = ""
"""



def process_file(input_file, output_file):
    all_single = True
    count_complex = 0
    with open(input_file, "r") as f, open(output_file, "w") as f_out:
        for idx, line in enumerate(f):
            instance = json.loads(line)
            # Add a new field in the instance to store the additionalsystem content
            instance["system_content"] = ""
            test_category = instance["id"].rsplit("_", 1)[0]
            if idx == 0:
                print(test_category)            

            if len(instance["question"]) == 1:
                # Just a single user command
                if len(instance["question"][0]) == 1:
                    # print(instance["question"][0])
                    assert instance["question"][0][0]["role"] == "user"
                    instance["question"] = instance["question"][0][0]["content"]
                elif len(instance["question"][0]) == 2:
                    # Mostly this instance is a system command + user command
                    if instance["question"][0][0]["role"] == "system" and instance["question"][0][1]["role"] == "user":
                        # Process system content which will be added to the system prompt
                        instance["system_content"] = instance["question"][0][0]["content"]
                        instance["question"] = instance["question"][0][1]["content"]
                    elif instance["question"][0][0]["role"] == "user" and instance["question"][0][1]["role"] == "user":
                        # Concatenate the two user commands - only happens for one instance
                        instance["question"] = (instance["question"][0][0]["content"] + instance["question"][0][1]["content"]).strip()
                    else:
                        raise ValueError(f"Unknown instance format: {json.dumps(instance, indent=4)}")
                else:
                    all_single = False
                    count_complex += 1
            else:
                all_single = False
                count_complex += 1

                
            if "function" in instance:
                instance["function"] = func_doc_language_specific_pre_processing(instance["function"], test_category)
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
        except Exception as e:
            print(f"Error: {e.message}")


if __name__ == "__main__":
    download_and_process_bfcl_data(
        REPO_URL, SUBFOLDER_PATH, 
        output_dir=os.path.join(os.path.dirname(__file__))
    )