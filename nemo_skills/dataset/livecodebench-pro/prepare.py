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

from datasets import load_dataset

if __name__ == '__main__':
    # Write an argparse to a json file, read it in and parse it
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = load_dataset("anonymous1926/anonymous_dataset")
    output_file_path = os.path.join(args.output_dir, f"livecodebench_pro.jsonl")
    with open(output_file_path, 'w') as f:
        for split_name, split in dataset.items():
            for row in split:
                row['task_id'] = row.pop('problem_id')
                row['question'] = row.pop('problem_statement')
                row['split'] = split_name
                json.dump(
                    row,
                    f,
                )
                f.write('\n')
