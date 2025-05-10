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
import os
import random
import tempfile

import wandb


def _process_and_log_samples(jsonl_file, num_samples, output_name, tmpdirname):
    with open(jsonl_file, 'r') as file:
        lines = list(enumerate(file.readlines(), 1))

    random_pairs = random.sample(lines, min(num_samples, len(lines)))
    samples_dict = {}

    for line_num, line in random_pairs:
        sample = json.loads(line)
        samples_dict[str(line_num)] = sample

    samples_file = os.path.join(tmpdirname, output_name)
    with open(samples_file, 'w') as f:
        json.dump(samples_dict, f, indent=2)

    wandb.save(samples_file, base_path=tmpdirname)
    wandb.summary["num_samples"] = len(lines)


def log_random_samples(jsonl_file, num_samples, project, name, group=None):
    # Initialize wandb
    wandb.init(
        project=project,
        name=name,
        id=name + (group if group else ""),
        resume="allow",
        group=group,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Process main samples
        _process_and_log_samples(jsonl_file, num_samples, "samples.json", tmpdirname)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log random samples from a JSONL file to wandb.")
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of random samples to log.")
    parser.add_argument("--project", type=str, default="nemo-skills", help="wandb project name.")
    parser.add_argument("--name", type=str, required=True, help="wandb run name.")
    parser.add_argument("--group", type=str, required=False, help="wandb group name.")

    args = parser.parse_args()

    log_random_samples(args.jsonl_file, args.num_samples, args.project, args.name, args.group)
