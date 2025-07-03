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

REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
)


def install_requirements(url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", url])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")


def install_or_upgrade_package(package_name):
    try:
        # Run the pip command to install or upgrade the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        print(f"{package_name} has been successfully installed or upgraded.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing/upgrading {package_name}: {e}")


# TODO: use sandbox
@nested_dataclass(kw_only=True)
class BigCodeBenchEvaluatorConfig:
    dataset: str = "bigcodebench"
    subset: str = "hard"


def eval_bigcodebench(cfg):
    try:
        from bigcodebench.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'bigcodebench' not found. Attempting to install...")
        install_requirements(REQUIREMENTS_URL)
        install_or_upgrade_package("bigcodebench")
        try:
            from bigcodebench.evaluate import evaluate
        except ImportError:
            LOG.info("Failed to install 'bigcodebench'. Please install it manually.")
            raise

    eval_config = BigCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    for jsonl_file in unroll_files(cfg.input_files):
        samples = []
        with open(jsonl_file) as f:
            for line in f:
                generation_dict = preprocess_code(json.loads(line))
                generation_dict["solution"] = generation_dict.pop("completion")
                samples.append(generation_dict)
        # all changes will be done with a new key "completion", so it's ok to write to the same file
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/evaluate.py#L117
        evaluate("instruct", eval_config.subset, samples=jsonl_file, execution="local")  # subset [full, hard]
        # if the input filename is "model-name--bigcodebench-instruct--sanitized.jsonl"
        # then there will be two output files (generated) after evluation:
        # "model-name--bigcodebench-instruct--sanitized_eval_results.json"
        # "model-name--bigcodebench-instruct--sanitized_pass_at_k.json"

        # moving eval file as otherwise bigcodebench does not want to recompute metrics if it's present..
        shutil.move(jsonl_file[:-6] + '_eval_results.json', jsonl_file[:-6] + '_eval_results-saved.json')
