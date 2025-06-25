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
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50


def test_code(model_name, split, log_dir, output_dir, with_background=False):
    # adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/test_generated_code.py

    with open('/home/igitman/workspace/NeMo-Skills/nemo_skills/dataset/scicode/test.jsonl') as fin:
        scicode_data = [json.loads(line) for line in fin]
    json_dct = {}
    json_idx = {}

    for prob_data in scicode_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = scicode_data.index(prob_data)
    start_time = time.time()

    code_dir_ = Path('/home/igitman/workspace/NeMo-Skills/tmp-scicode-dir')
    tmp_dir = Path(f'tmp_{start_time}')

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for file_path in code_dir_.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = file_path.read_text(encoding='utf-8')
            json_content = scicode_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(
                    """

import h5py
import scipy
H5PY_FILE = "/home/igitman/workspace/NeMo-Skills/test_data.h5"

def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file=H5PY_FILE):
    data_lst = []
    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst



"""
                )
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

    for file_path in tmp_dir.iterdir():
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split('.')[0]
            print(f'Testing function {func_id} ...')
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = Path(log_dir, model_name)
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

    print(
        f'correct problems: {correct_prob_num}/{DEV_PROB_NUM if (split == "validation") else PROB_NUM - DEV_PROB_NUM}'
    )
    print(f'correct steps: {len(correct_step)}/{DEV_STEP_NUM if (split == "validation") else STEP_NUM}')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir}/{model_name}.txt', 'w') as f:
        f.write(
            f'correct problems: {correct_prob_num}/{DEV_PROB_NUM if (split == "validation") else PROB_NUM - DEV_PROB_NUM}\n'
        )
        f.write(f'correct steps: {len(correct_step)}/{DEV_STEP_NUM if (split == "validation") else STEP_NUM}\n\n')
        f.write(f'duration: {test_time} seconds\n')
        f.write('\ncorrect problems: ')
        f.write(f'\n\n{[i + 1 for i in range(PROB_NUM) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0]}\n')

    with open(f'{output_dir}/{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(correct_dict, f, indent=4)

    shutil.rmtree(tmp_dir)


def eval_scicode(cfg):
    test_code('model', 'test', 'tmp-scicode-eval', 'tmp-scicode-eval')
    # for file in unroll_files(cfg.input_files):
    #     with open(file, 'rt', encoding='utf-8') as fin:
    #         data = [json.loads(line) for line in fin]
    #     with open(file, 'wt', encoding='utf-8') as fout:
    #         for sample in tqdm(data):
    #             sample['predicted_answer'] = extract_answer(sample["generation"])
    #             sample['is_correct'] = sample['predicted_answer'] == sample['expected_answer']
    #             fout.write(json.dumps(sample) + "\n")
