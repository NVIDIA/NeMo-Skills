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

import argparse
import json


def filter(input_file, output_file_yes, output_file_no, mode):
    with open(input_file, 'r') as infile, open(output_file_yes, 'w') as outfile_yes, open(
        output_file_no, 'w'
    ) as outfile_no:
        for line in infile:
            data = json.loads(line)
            generation = data.pop("generation")
            data[f"classify_{mode}_gen"] = generation
            if generation.startswith(mode):
                outfile_yes.write(json.dumps(data) + '\n')
            else:
                outfile_no.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Will only keep proof questions based on the expected_answer field.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file_yes", help="Path to the output JSONL file")
    parser.add_argument("output_file_no", help="Path to the output JSONL file")
    parser.add_argument("--mode", required=True, help="Classification mode")

    args = parser.parse_args()

    filter(args.input_file, args.output_file_yes, args.output_file_no, args.mode)
