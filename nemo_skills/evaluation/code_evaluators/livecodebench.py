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
import sys

from nemo_skills.evaluation.code_utils import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def install_from_git(git_url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
        print("Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")


# TODO: use sandbox
@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    dataset: str = "livecodebench"
    language: str = "python"  # "cpp" is another option now
    release_version: str = "v5"
    test_file: str = None


def eval_livecodebench(cfg):
    try:
        from livecodebench.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'livecodebench' not found. Attempting to install...")
        install_from_git("git+https://github.com/wasiahmad/livecodebench.git")
        try:
            from livecodebench.evaluate import evaluate
        except ImportError:
            LOG.info("Failed to install 'livecodebench'. Please install it manually.")
            raise

    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)
    assert eval_config.language in ["python", "cpp"]
    if eval_config.language == "cpp":
        assert eval_config.test_file is not None

    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line), eval_config.language) for line in f]
            for sample in samples:
                sample["question_id"] = sample["task_id"]
                sample["code_list"] = [sample["completion"]]
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # https://github.com/wasiahmad/livecodebench/blob/main/livecodebench/evaluate.py#L10
        evaluate(
            custom_output_file=jsonl_file,
            release_version=f"release_{eval_config.release_version}",
            k_list=[1],
            language=eval_config.language,
            test_file=None if eval_config.language == "python" else eval_config.test_file,
            num_process_evaluate=12,
            timeout=6 if eval_config.language == "python" else 30,
        )

        with open(jsonl_file[:-6] + '_eval_results.json', 'rt', encoding="utf-8") as fin:
            eval_grades = json.load(fin)
        # adding is_correct key to allow compute_metrics to work
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample['graded_list'] = eval_grades['eval'][sample['task_id']]['graded_list']
                f.write(json.dumps(sample) + "\n")

        # moving eval file to ensure metrics are recomputed
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')
