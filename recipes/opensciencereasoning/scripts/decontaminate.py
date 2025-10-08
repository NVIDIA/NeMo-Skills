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
import json
import argparse
import logging

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Decontaminate the input file."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="file to decontaminate",
    )
    parser.add_argument(
        "-d", "--dec_path", required=True, help="file with the 'contaminated' field"
    )
    parser.add_argument(
        "-s", "--save_path", required=True, help="save path"
    )
    parser.add_argument(
        "-f", "--with_duplicates", default=False, help="flag to leave duplicated problems"
    )
    args = parser.parse_args()

    dec = set()

    with open(args.dec_path) as fin:
        for line in fin:
            sample = json.loads(line)
            if not sample['contaminated']:
                dec.add(sample['problem'])

    with open(args.input_path) as fin, open(args.save_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            if sample['problem'] in dec:
                if not args.with_duplicates:
                    dec.remove(sample['problem'])
                fout.write(line)


if __name__ == "__main__":
    main()
