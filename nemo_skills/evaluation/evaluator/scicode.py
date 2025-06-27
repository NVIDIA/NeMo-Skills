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
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.eval.scicode_utils import eval_prefix
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ScicodeEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'local'})
    timeout: float = 30.0
    num_parallel_requests: int = 20


def _execute_single_test(args):
    """Helper function to execute a single test case."""
    eval_config, elem_idx, full_generation, json_content, subtask_step = args

    # step_id is always problem_id.subtask_step
    step_number = json_content["sub_steps"][int(subtask_step) - 1]["step_number"]
    test_lst = json_content["sub_steps"][int(subtask_step) - 1]["test_cases"]
    code = full_generation + eval_prefix + f"targets = process_hdf5_to_tuple('{step_number}', {len(test_lst)})\n"
    for idx in range(len(test_lst)):
        code += f"target = targets[{idx}]\n\n"
        for line in test_lst[idx].split('\n'):
            code += line + '\n'

    sandbox = get_sandbox(**eval_config.sandbox)
    output_dict, _ = sandbox.execute_code(code, timeout=eval_config.timeout, max_output_characters=100000)

    return elem_idx, output_dict


def test_code(eval_config, scicode_data):
    # adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/test_generated_code.py
    json_idx = {}

    for prob_data in scicode_data:
        json_idx[prob_data['problem_id']] = scicode_data.index(prob_data)

    # Prepare all tasks for parallel execution
    tasks = []
    for elem_idx, elem in enumerate(scicode_data):
        for step_id, full_generation in elem['generation'].items():
            problem_id, subtask_step = step_id.split('.')
            json_content = scicode_data[json_idx[problem_id]]
            tasks.append((eval_config, elem_idx, full_generation, json_content, subtask_step))

    # Initialize status_lists with correct structure
    status_lists = [[] for _ in range(len(scicode_data))]

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=eval_config.num_parallel_requests) as executor:
        results = list(executor.map(_execute_single_test, tasks))

    # Organize results back into the original structure
    for elem_idx, output_dict in results:
        status_lists[elem_idx].append(output_dict)

    return status_lists


def eval_scicode(cfg):
    eval_config = ScicodeEvaluatorConfig(**cfg.eval_config)
    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]
        status_lists = test_code(eval_config, data)
        with open(file, 'wt', encoding='utf-8') as fout:
            for idx, elem in enumerate(data):
                elem['eval_status'] = status_lists[idx]
                fout.write(json.dumps(elem) + "\n")
