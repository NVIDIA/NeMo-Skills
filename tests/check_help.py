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

# a helper script to check that running --help works on all scripts
# not launching via pytest since it needs a different environment
# (no packages installed) than other tests

import argparse
import subprocess
import sys

pipeline_script_list = [
    'pipeline/compute_metrics.py',
    'pipeline/prepare_eval.py',
    'pipeline/run_eval.py',
    'pipeline/run_labeling.py',
    'pipeline/run_pipeline.py',
    'pipeline/run_training.py',
    'pipeline/start_server.py',
    'pipeline/summarize_results.py',
]

# only testing scripts that don't require nemo/trtllm to avoid setting up dockers
skills_script_list = [
    'nemo_skills/inference/generate_solutions.py',
    'nemo_skills/evaluation/evaluate_results.py',
    'nemo_skills/evaluation/fill_majority_answer.py',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all scripts')
    args = parser.parse_args()

    script_list = pipeline_script_list
    if args.all:
        script_list += skills_script_list

    for script in script_list:
        cmd = f'{sys.executable} {script} --help'
        print(f'Running {cmd}')
        subprocess.run(cmd, shell=True, check=True)
