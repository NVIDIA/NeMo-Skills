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
import logging

def check_topic_structure(sample: dict, topics_structure: dict, names: list):
    """Validate that each hierarchical label in `sample` is allowed by `topics_structure`.

    Expected `topics_structure` shape for two levels:
      {
        "topics": ["Chemistry", "Physics", ...],
        "subtopics": {"Chemistry": [..], "Physics": [..], ...}
      }
    This function also works if deeper levels follow the same pattern: each level
    after the first is a dict keyed by the previous level's selected value.
    """
    if not names:
        return True

    # First level is a flat list of allowed values
    first_name = names[0]
    first_allowed = topics_structure.get(first_name, [])
    if sample.get(first_name) not in first_allowed:
        return False

    # Each subsequent level is a mapping from the previous selection to allowed values
    for i in range(1, len(names)):
        prev_name = names[i - 1]
        curr_name = names[i]
        prev_value = sample.get(prev_name)
        curr_value = sample.get(curr_name)

        mapping_or_list = topics_structure.get(curr_name, {})
        if isinstance(mapping_or_list, dict):
            allowed = mapping_or_list.get(prev_value, [])
        else:
            allowed = mapping_or_list

        if curr_value not in allowed:
            return False

    return True

def aggregate_topics(input_files: dict, output_file: str, topics_structure: dict, names: list):
    """Merge per-level classification outputs into a single JSONL.

    - `input_files`: mapping from label key (e.g., "topics") to its output.jsonl
    - `topics_structure`: control structure used to filter invalid hierarchical pairs
    - `names`: ordered list of keys defining the hierarchy (e.g., ["topics", "subtopics"]).
    """
    data = {}
    for topic_key, file in input_files.items():
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample['problem'] not in data:
                    data[sample['problem']] = sample
                data[sample['problem']][topic_key] = sample[topic_key]
    with open(output_file, "w") as f:
        for sample in data.values():
            if check_topic_structure(sample, topics_structure, names):
                f.write(json.dumps(sample) + "\n")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Aggregate topics.")
    parser.add_argument("--input_files", required=True, type=json.loads)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--topics_structure", default=None, type=json.loads)
    parser.add_argument("--names", default=None, type=json.loads)
    args = parser.parse_args()
    aggregate_topics(args.input_files, args.output_file, args.topics_structure, args.names)

if __name__ == "__main__":
    main()