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
import re
import shutil
from argparse import Namespace
from dataclasses import field

from omegaconf import OmegaConf

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def preprocess_code(generation_dict: dict, language="python"):
    completion = generation_dict['generation']
    completion = completion.strip()
    completion = completion.replace("\r", "")

    ##### To handle code generation by reasoning models
    # check for <think> and </think> tags
    if "<think>" in completion:
        if "</think>" in completion:
            # thinking trace completed, solution in after the trace
            match = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
            completion = match.group(1).strip() if match else None
        else:
            completion = None

    if completion is None:
        generation_dict["completion"] = ""  # no valid solution generated
        return generation_dict
    #####

    start_with_lang_tag = f'```{language}'
    generic_start_end_tag = f'```'

    if start_with_lang_tag in completion:
        def_line = completion.index(start_with_lang_tag) + len(start_with_lang_tag)
        completion = completion[def_line:].strip()
        try:
            next_line = completion.index(generic_start_end_tag)
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")

    elif generic_start_end_tag in completion:
        def_line = completion.index(generic_start_end_tag) + len(generic_start_end_tag)
        completion = completion[def_line:].strip()
        try:
            next_line = completion.index(generic_start_end_tag)
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")

    if completion.startswith(" "):
        completion = completion.strip()

    generation_dict["completion"] = completion
    return generation_dict


@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'local'})
    language: str = "python"  # "cpp" is another option now
    test_file: str = None
    timeout: float = 6.0


def eval_livecodebench(cfg):
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)
    assert eval_config.language in ["python"]  # cpp language support is pending

    release_version = None
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line), eval_config.language) for line in f]
            for sample in samples:
                sample["code_list"] = [sample["completion"]]
                if release_version is None:
                    release_version = sample["release_version"]
                if release_version != sample["release_version"]:
                    raise ValueError(
                        f"All samples should have the same release version, "
                        f"but got {release_version} and {sample['release_version']}"
                    )

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        code = f"""
import sys
import subprocess

def install_from_git(git_url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
        print("Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {{e}}")
        
try:
    from livecodebench.evaluate import evaluate
except ImportError:
    print("Package 'livecodebench' not found. Attempting to install...")
    install_from_git("git+https://github.com/wasiahmad/livecodebench.git")
    try:
        from livecodebench.evaluate import evaluate
    except ImportError:
        print("Failed to install 'livecodebench'. Please install it manually.")
        raise

evaluate(
    custom_output_file={jsonl_file},
    test_file={eval_config.test_file},
    k_list=[1],
    language={eval_config.language},
    num_process_evaluate=12,
    timeout={eval_config.timeout},
)
"""
        sandbox = get_sandbox(**eval_config.sandbox)
        output_dict, _ = sandbox.execute_code(
            code, timeout=eval_config.timeout * len(samples), max_output_characters=100000
        )

        with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
            eval_grades = json.load(fin)
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample['graded_list'] = eval_grades['eval'][sample['task_id']]['graded_list']
                f.write(json.dumps(sample) + "\n")

        # moving eval file to ensure metrics are recomputed
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')


def eval_livecodebench_pro(cfg):
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line), "python") for line in f]
            for sample in samples:
                sample["problem_id"] = sample.pop("task_id")
                sample["text_response"] = sample.pop("completion")
                sample["response_meta"] = None

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")


def eval_evalplus(cfg):
    # TODO: need to move it to a separate docker (either our sandbox or separate srun)
    from evalplus.evaluate import evaluate

    # processing each generation separately (TODO: evalplus can do it together, but need to figure out the format)
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line)) for line in f]
        # all changes will be done with a new key "completion", so it's ok to write to the same file
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        eval_config = {
            "samples": jsonl_file,
            "base_only": False,
            "parallel": None,
            "i_just_wanna_run": False,
            "test_details": False,
            "min_time_limit": 1,
            "gt_time_limit_factor": 4.0,
            "mini": False,
            "noextreme": False,
            "version": "default",
        }
        eval_config.update(OmegaConf.to_container(cfg.eval_config))
        evaluate(Namespace(**eval_config))
        with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
            evalplus_grades = json.load(fin)
        # adding is_correct key to allow compute_metrics to work
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample['is_correct'] = evalplus_grades['eval'][sample['task_id']][0]['base_status'] == "pass"
                sample['is_correct-plus'] = (
                    sample['is_correct'] and evalplus_grades['eval'][sample['task_id']][0]['plus_status'] == "pass"
                )
                f.write(json.dumps(sample) + "\n")

        # moving eval file as otherwise evalplus does not want to recompute metrics if it's present..
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')
