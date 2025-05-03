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

"""Selects the best summary from the new summaries and makes that the GenSelect target output."""


import argparse
import json
import glob
from os import path

from utils import extract_judgment


def read_jsonl_file(file_path, key=None):
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            if key is not None:
                instances.append(instance[key])
            else:
                instances.append(instance)
    
    return instances


def is_valid_summary(reasoning_judgment, summary_generation):
    """Check if the summary is valid, i.e. the summary judgment is the same as the reasoning judgment"""
    summary_judgment = extract_judgment(summary_generation)
    if summary_judgment is None:
        return False
    return (reasoning_judgment == summary_judgment)


def select_best_summary(valid_summaries):
    """Select the best summary from the valid summaries"""
    # Currently we just select the longest valid summary
    return max(valid_summaries, key=lambda x: len(x["generation"]))
    

def format_reasoning_trace_with_summary(reasoning_file, summary_dir):
    """Selects the best summary from the new summaries and makes that the GenSelect target output."""
    reasoning_instances = read_jsonl_file(reasoning_file)
    # We have multiple summaries for the same reasoning trace
    list_of_summary_instances = [read_jsonl_file(summary_file) for summary_file in glob.glob(path.join(summary_dir, "*.jsonl"))]

    # The number of reasoning traces should match the number of summaries for all the summary files
    # assert all(len(reasoning_instances) == len(summary_instances) for summary_instances in list_of_summary_instances)
    list_of_summary_instances = [summary_instances for summary_instances in list_of_summary_instances if len(reasoning_instances)==len(summary_instances)]

    if len(list_of_summary_instances) == 0:
        return [], 0, len(reasoning_instances)

    formatted_instances = []
    valid_summary_count = 0
    invalid_summary_count = 0

    all_summaries = list(zip(*list_of_summary_instances))
    for idx, (reasoning_instance, summaries_for_reasoning_instance) in enumerate(zip(reasoning_instances, all_summaries)):
        reasoning_judgment = reasoning_instance["judgment"]
        assert (reasoning_instance["problem"] == summaries_for_reasoning_instance[0]["problem"])
        valid_summaries = [summary for summary in summaries_for_reasoning_instance if is_valid_summary(reasoning_judgment, summary["generation"])]

        if len(valid_summaries) == 0:
            invalid_summary_count += 1
            continue 
        else:
            best_summary = select_best_summary(valid_summaries)
            reasoning_instance["generation"] = best_summary["generation"]            
            valid_summary_count += 1
            formatted_instances.append(reasoning_instance)

    return formatted_instances, valid_summary_count, invalid_summary_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_file", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    formatted_instances, _, _ = format_reasoning_trace_with_summary(args.reasoning_file, args.summary_dir)
    with open(args.output_file, "w") as f:
        for instance in formatted_instances:
            f.write(json.dumps(instance) + "\n")
    

if __name__ == "__main__":
    main()
