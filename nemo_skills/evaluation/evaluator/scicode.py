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
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.inference.eval.scicode_utils import eval_prefix
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50


def test_code(scicode_data):
    # adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/test_generated_code.py
    json_dct = {}
    json_idx = {}

    for prob_data in scicode_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = scicode_data.index(prob_data)
    start_time = time.time()

    tmp_dir = Path(f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    total_problems = len(scicode_data)
    total_steps = 0

    for elem in scicode_data:
        for step_id, full_generation in elem['generation'].items():
            problem_id, subtask_step = step_id.split('.')
            total_steps += 1
            code_content = file_path.read_text(encoding='utf-8')
            json_content = scicode_data[json_idx[problem_id]]
            step_id = json_content["sub_steps"][int(subtask_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(subtask_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(eval_prefix)
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')

    def run_script(script_path):
        script_path = str(script_path)
        try:
            subprocess.run(['python', script_path], check=True, text=True, timeout=1800)
            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error running script {script_path}: {e}")
            print(e)
            return 1
        except subprocess.TimeoutExpired as e:
            print(f"Runtime error while running script {script_path}: {e}")
            return 2

    correct_prob = np.zeros(PROB_NUM)
    tot_prob = np.zeros(PROB_NUM)
    correct_step = []
    correct_dict = {}

    for i in range(PROB_NUM):
        correct_dict[f'{i+1}'] = []

    log_dir = '/workspace/NeMo-Skills/tmp-scicode-logs'
    for file_path in tmp_dir.iterdir():
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split('.')[0]
            print(f'Testing function {func_id} ...')
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = Path(log_dir)
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir_, f'{file_path.stem}.txt')
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        correct_prob[int(prob_id) - 1] += 1
                        correct_step.append(func_id)
                        correct_dict[prob_id].append(func_id)
                continue
            ret = run_script(file_path)
            if ret == 0:
                correct_prob[int(prob_id) - 1] += 1
                correct_step.append(func_id)
                correct_dict[str(prob_id)].append(func_id)
                with open(logs_file, 'w') as f:
                    f.write('pass')
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')

    test_time = time.time() - start_time

    correct_prob_num = sum(1 for i in range(PROB_NUM) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0)

    split = 'test'
    print(total_problems, total_steps)
    print(
        f'correct problems: {correct_prob_num}/{DEV_PROB_NUM if (split == "validation") else PROB_NUM - DEV_PROB_NUM}'
    )
    print(f'correct steps: {len(correct_step)}/{DEV_STEP_NUM if (split == "validation") else STEP_NUM}')

    shutil.rmtree(tmp_dir)


def eval_scicode(cfg):
    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]
        test_code(data)
