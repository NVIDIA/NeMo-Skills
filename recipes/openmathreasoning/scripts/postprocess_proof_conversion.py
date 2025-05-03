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
import glob
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", help="Glob pattern for the input JSONL files")
    parser.add_argument("output_file", help="Path to the output JSONL file")

    args = parser.parse_args()

    with open(args.output_file, 'w') as outfile:
        for input_file in glob.glob(args.input_files):
            with open(input_file, 'r') as infile:
                for line in infile:
                    data = json.loads(line)
                    # there should not be any expected answer, but dropping it just in case
                    data["expected_answer"] = None
                    data["original_problem"] = data.pop("problem")
                    data["problem"] = data.pop("generation")
                    outfile.write(json.dumps(data) + '\n')
