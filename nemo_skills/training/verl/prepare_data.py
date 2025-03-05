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
import pandas as pd
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet with specific transformations.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output Parquet file.')
    parser.add_argument('--data_source', type=str, default='nemo-skills', help='Data source to be recorded in the output.')
    parser.add_argument('--ability', type=str, default='math', help='Ability to be recorded in the output.')
    return parser.parse_args()

def transform_data(input_file, data_source, ability):
    # Read the JSONL file
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            transformed_entry = {
                'prompt': json_line['input'],
                'reward_model': {
                    'ground_truth': json_line['expected_answer'],
                    'style': 'rule'
                },
                'extra_info': {
                    'problem': json_line['problem']
                },
                'data_source': data_source,
                'ability': ability
            }
            data.append(transformed_entry)

    # Convert list to DataFrame
    df = pd.DataFrame(data)
    return df

def save_to_parquet(df, output_file):
    df.to_parquet(output_file, index=False)

def main():
    args = parse_args()
    transformed_df = transform_data(args.input_file, args.data_source, args.ability)
    save_to_parquet(transformed_df, args.output_file)
    print(f"Data transformed and saved to {args.output_file}")

if __name__ == "__main__":
    main()

