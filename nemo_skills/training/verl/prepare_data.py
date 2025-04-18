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
import pandas as pd
import uuid

def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet with specific transformations.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the the list of jsonl files.')
    parser.add_argument('--global_step', type=str, required=True, help='Path to save the jsonl file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output Parquet file.')
    parser.add_argument('--data_source', type=str, default='nemo-skills', help='Data source to be recorded in the output.')
    parser.add_argument('--ability', type=str, default='math', help='Ability to be recorded in the output.')
    return parser.parse_args()



def interleave_jsonl_files(folder_path, global_step):
    # Get a sorted list of all .jsonl files in the folder
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.jsonl')])

    if not file_list:
        raise ValueError("No .jsonl files found in the folder")

    all_contents = []
    # Read all lines from each .jsonl file
    for filename in file_list:
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            all_contents.append(lines)

    # Ensure all files have the same number of lines
    num_lines = len(all_contents[0])
    if not all(len(lines) == num_lines for lines in all_contents):
        raise ValueError("All .jsonl files must have the same number of lines")

    # Interleave lines: take line 1 from each file, then line 2 from each, etc.
    merged_lines = []
    for i in range(num_lines):
        for lines in all_contents:
            merged_lines.append(lines[i])

    # Write the interleaved result to the output .jsonl file
    
    output_path = os.path.join(folder_path, f"{global_step}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in merged_lines:
            f.write(line + '\n')

    return output_path


def transform_data(input_file, data_source, ability):
    # Read the JSONL file and transform each entry
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            transformed_entry = {
                # Format the prompt as a list with role and content
                'prompt': [
                    {
                        'content': "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n" + json_line['problem'],
                        'role': 'user'
                    }
                ],
                # Provide the expected answer and reward style
                'reward_model': {
                    'ground_truth': json_line['expected_answer'],
                    'style': 'rule-lighteval/MATH_v2'
                },
                'response': json_line['response'],
                # Include extra info such as a unique index
                'extra_info': {
                    'index': str(uuid.uuid4()),
                    'problem': json_line['problem'],
                    'regex': '\\\\boxed\\s*{\\s*(.+?)\\s*}',
                },
                # Metadata: source and type of ability tested
                'data_source': data_source,
                'ability': ability
            }
            data.append(transformed_entry)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    return df


def save_to_parquet(df, output_file):
    df.to_parquet(output_file, index=False)

def main():
    args = parse_args()
    output_path = interleave_jsonl_files(args.input_folder, args.global_step)
    args.input_file = output_path
    transformed_df = transform_data(args.input_file, args.data_source, args.ability)
    save_to_parquet(transformed_df, args.output_file)
    print(f"Data transformed and saved to {args.output_file}")

if __name__ == "__main__":
    main()

