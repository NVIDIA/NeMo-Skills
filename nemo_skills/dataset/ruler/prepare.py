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

import argparse
import json
import os
from pathlib import Path
import glob
from shutil import copyfile
from copy import deepcopy
# prepare ruler jsons from steps: 

# Define the configuration as a dictionary
default_config = {
    "PROMPT_CONFIG": "generic/default",
    "DATASET_GROUP": "chat",
    "METRICS_TYPE": "ruler",
    "DEFAULT_EVAL_ARGS": {
        "eval_type": "ruler",
        "eval_config.match_type": "all"
    },
    "DEFAULT_GENERATION_ARGS": {
        "inference.tokens_to_generate": 120,
        "input_file": "/ruler/ruler_llama_4k/niah-single_1/test.jsonl", # example. will be overwritten
        "dataset": 'null', # overwrite dataset and split to use input_file
        "split": 'null'
    }
}

# Define the tokens_to_generate for each task from RULER/scripts/data/synthetic/constants.py
tokens_to_generate ={
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

def write_config_to_file(file_path, config):
    """
    Writes the configuration dictionary to a file in the desired format.
    
    Args:
        file_path (str): Path to the file.
        config (dict): Configuration dictionary to write.
    """
    def format_value(key, value):
        """Format the value for the output."""
        if isinstance(value, dict):
            # Format nested dictionaries as key-value pairs with "++"
            return " ".join(
                f"++{sub_key}={sub_value}" for sub_key, sub_value in value.items()
            )
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return value

    with open(file_path, "a") as file:
        for key, value in config.items():
            if isinstance(value, dict):
                # Special formatting for dictionary values like DEFAULT_EVAL_ARGS
                formatted_value = format_value(key, value)
                file.write(f"{key} = \"{formatted_value}\"\n")
            else:
                # Write simple key-value pairs
                formatted_value = format_value(key, value)
                file.write(f"{key} = {formatted_value}\n")

def update_config_for_task(config, eval_suite, task, subset_file):
    # update the config for the task
    short_task_name = task.split('_')[0]
    updated_config = deepcopy(config)
    updated_config['DEFAULT_GENERATION_ARGS']['inference.tokens_to_generate'] = tokens_to_generate[short_task_name]
    updated_config['DEFAULT_GENERATION_ARGS']['input_file'] = f"/ruler/{eval_suite}/{task}/{subset_file}"

    if short_task_name == 'qa':
        updated_config['DEFAULT_EVAL_ARGS']['eval_config.match_type'] = 'part'
   
    return updated_config


def process_one_file(original_file, output_json_suite_folder, dataset_folder, eval_suite):
    task, subset_file =  original_file.split("/")[-2], original_file.split("/")[-1]
   
    # create jsonl task folder in args.output_json_folder
    output_json_task_folder = Path(f"{output_json_suite_folder}/{task}")
    output_json_task_folder.mkdir(exist_ok=True)
    # create jsonl file in output_json_task_folder
    output_json_file = os.path.join(output_json_task_folder, subset_file)
    task_folder = Path(f"{dataset_folder}/{task}")  
    task_folder.mkdir(exist_ok=True)

    updated_config = update_config_for_task(default_config, eval_suite, task, subset_file)
    # copy the config file #TODO: can we remove the license title if not published?
    copyfile("./__init__.py", os.path.join(task_folder, "__init__.py"))
    # write the config to the file
    write_config_to_file(os.path.join(task_folder, "__init__.py"), updated_config)

    with open(original_file, "r") as fin, open(output_json_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            original_entry = json.loads(line)
            new_entry = dict(
                index=original_entry["index"],
                problem=original_entry["input"],
                expected_answer=original_entry["outputs"],
                length=original_entry["length"],
                assistant_prefix=original_entry['answer_prefix']
            )
            
            fout.write(json.dumps(new_entry) + "\n")

if __name__ == "__main__":
    """
    python prepare.py \
    --input_json_folders /data2/long_context_eval/ruler/original/base_*/ /data2/long_context_eval/ruler/original/llama_*/ \
    --output_json_folder /data2/long_context_eval/ruler/ns/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_folders", type=str, nargs='+', required=True, 
                       help="input jsonl folders (supports multiple arguments and glob patterns)")
    parser.add_argument("--output_json_folder", type=str, required=True, help="where to save the output ns style jsonl files")
    args = parser.parse_args()

    #Find all input_json_folder
    input_folders = []
    for pattern in args.input_json_folders:
        matched_folders = glob.glob(pattern)
        if not matched_folders:
            print(f"Warning: No folders found matching pattern: {pattern}")
        input_folders.extend(matched_folders)
    if not input_folders:
        raise ValueError("No folders found matching any of the provided patterns")

    for input_json_folder in input_folders:
        print(f"Processing {input_json_folder}")
        original_files = [f for f in glob.glob(f"{input_json_folder}/**", recursive=True) if os.path.isfile(f)]
    
        if len(original_files) == 0:   
            raise ValueError("No original files found. please check the --input_json_folders")
        else:
            # extract the eval_suite name from the input_json_folder
            eval_suite = original_files[0].split("/")[-3].split("_original")[0]
            # create eval suite folder in output_json_folder
            output_json_suite_folder = Path(f"{args.output_json_folder}/{eval_suite}")
            output_json_suite_folder.mkdir(parents=True, exist_ok=True)
            # create eval suite in nemo-skills/dataset/ruler/
            dataset_folder = Path(f"./{eval_suite}")
            dataset_folder.mkdir(parents=True, exist_ok=True)

        for original_file in original_files:
            print(f"Processing {original_file}")
            process_one_file(original_file, output_json_suite_folder, dataset_folder, eval_suite)

        print(f"Done. Saved eval sutie {eval_suite} jsonl files to {args.output_json_folder}")
        print(f"""     ==================================\n \
        You can upload the jsonl files to cluster and skip uploading the dataset folder \n \
        Or we have pre-generated jsonl files on most cluster. \n\n \
        Make sure to mount the jsonl folder to /ruler/ in cluster config yaml files!\n \
        i.e. /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_long-context/eval/ruler_ns/:/ruler/\n \
        ==================================
        """)