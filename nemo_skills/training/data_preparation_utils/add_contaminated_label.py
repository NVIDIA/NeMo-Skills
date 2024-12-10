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
from pathlib import Path

from nemo_skills.utils import unroll_files


def load_contaminated_problems(jsonl_file):
    contaminated_problems = set()
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['contaminated']:
                contaminated_problems.add(data['problem'])
    return contaminated_problems


def update_output_files(data_files, contaminated_problems):
    for file_path in unroll_files(data_files):
        temp_file_path = Path(file_path).with_suffix('.temp')

        with open(file_path, 'r') as input_file, open(temp_file_path, 'w') as output_file:
            for line in input_file:
                data = json.loads(line)
                if data['problem'] in contaminated_problems:
                    data['contaminated'] = True
                json.dump(data, output_file)
                output_file.write('\n')

        # Replace the original file with the updated one
        temp_file_path.replace(file_path)
        print(f"Updated file: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Add contamination labels to problem files')
    parser.add_argument(
        '--label_file', type=str, required=True, help='Path to the file containing contaminated labels'
    )
    parser.add_argument(
        '--data_files', type=str, nargs='+', required=True, help='Glob pattern(s) for the files to update'
    )

    args = parser.parse_args()

    contaminated_problems = load_contaminated_problems(args.label_file)
    update_output_files(args.data_files, contaminated_problems)


if __name__ == '__main__':
    main()
