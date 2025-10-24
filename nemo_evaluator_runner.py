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

import subprocess
import tempfile
from pathlib import Path

from nemo_evaluator_launcher.api import RunConfig
from nemo_evaluator_launcher.common.helpers import (
    get_eval_factory_command,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_from_mapping,
    load_tasks_mapping,
)
from omegaconf import OmegaConf

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig


def get_cfg():
    yml_file = r"""
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: llama_3_1_8b_instruct_results
  # mode: sequential  # enables sequential execution

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NVIDIA_API_KEY # API Key with access to build.nvidia.com

# specify the benchmarks to evaluate
evaluation:
  nemo_evaluator_config:  # global config settings that apply to all tasks
    config:
      output_dir: /results_overriden
      params:
        request_timeout: 3600  # timeout for API requests in seconds
        parallelism: 5  # number of parallel requests
        limit_samples: 5

      target:
        api_endpoint:
          adapter_config:
            use_response_logging: true
            use_request_logging: true
    target:
      api_endpoint:
        api_key: API_KEY_BOO

  tasks:
    - name: aime_2025_nemo
      nemo_evaluator_config:  # task-specific configuration
        config:
          params:
            temperature: 0.6  # sampling temperature
            top_p: 0.95  # nucleus sampling parameter
            max_new_tokens: 8192  # maximum tokens to generate

    """
    folder = tempfile.mkdtemp()
    with open(Path(folder) / "config.yaml", "w") as f:
        f.write(yml_file)

    # WIPP: this is hardcoded config
    cfg = RunConfig.from_hydra(config_dir=folder, config_name="config")

    logger.info("Created launcher config", cfg=cfg)
    return cfg


class NemoEvaluatorRunner(GenerationTask):
    def __init__(self):
        pass

    def generate(self):
        logger.info("Getting tasks mapping")
        tasks_mapping = load_tasks_mapping()
        # WIPP: this is hardocoded task
        task_definition = get_task_from_mapping("aime_2025_nemo", tasks_mapping)
        logger.info("Loaded task definition", task=task_definition)
        cfg = get_cfg()
        eval_factory_command_struct = get_eval_factory_command(cfg, cfg.evaluation.tasks[0], task_definition)
        eval_factory_command = eval_factory_command_struct.cmd
        eval_factory_command_debug_comment = eval_factory_command_struct.debug
        logger.info("Generated command", cmd=eval_factory_command, debug=eval_factory_command_debug_comment)
        try:
            process = subprocess.Popen(eval_factory_command, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1)

            for line in process.stdout:
                print(line.strip())
            process.stdout.close()
            process.wait()
        except subprocess.SubprocessError as e:
            logger.error("Error running command", cmd=eval_factory_command, err=e)
            raise


GENERATION_TASK_CLASS = NemoEvaluatorRunner


if __name__ == "__main__":
    nemo_evaluator_runner = NemoEvaluatorRunner()
    nemo_evaluator_runner.generate()
