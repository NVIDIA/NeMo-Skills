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

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

import hydra
import json
from pathlib import Path
import re   
import shutil

from nemo_skills.inference.eval.bfcl_utils import extract_tool_response
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLGenerationConfig(GenerateSolutionsConfig):
    """BFCL benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "eval/bfcl/nemotron"
    prompt_template: str = "llama3-instruct"

    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"
    remove_thinking: bool = True

    tool_call_start_token = "<TOOLCALL>"
    tool_call_end_token = "</TOOLCALL>"


    @property
    def tool_call_regex(self):
        """Compiled regex pattern for extracting tool calls."""
        return re.compile(
            r"{}(.*?){}".format(
                re.escape(self.tool_call_start_token), 
                re.escape(self.tool_call_end_token)
            ), 
            re.DOTALL
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_bfcl_generation_config", node=BFCLGenerationConfig)


class BFCLGenerationTask(GenerationTask):
    def __init__(self, cfg: BFCLGenerationConfig):
        super().__init__(cfg)

        if not self.use_async_loop:  # if it was True, this message is printed by base class
            LOG.info(
                "Async loop is maintaining %d generations in parallel. "
                "Use max_concurrent_requests to control the number of concurrent requests.",
                self.cfg.max_concurrent_requests,
            )
            if self.server["server_type"] in ["nemo", "megatron"] and self.prompt_template is None:
                LOG.warning(
                    "NeMo/Megatron servers don't support inflight batching, "
                    "but SciCode evaluation requires it for efficient inference. "
                    "Each request will be processed 1 by 1, which is extremely inefficient and slow! "
                    "We highly recommend switching to a server that supports inflight batching."
                )
        self.use_async_loop = True  # SciCode is a multi-call benchmark, so we have to use async loop


    def postprocess(self):
        # Extract the tool response from the generation
        print("Postprocessing {}".format(self.cfg.output_file))
        temp_file = Path(self.cfg.output_file).with_suffix(".tmp")

        with open(self.cfg.output_file, "rt", encoding="utf-8") as fin, open(temp_file, "wt", encoding="utf-8") as fout:
            for line in fin:
                instance = json.loads(line)
                extracted_tool_response = extract_tool_response(
                    instance["generation"], self.cfg.tool_call_start_token, self.cfg.tool_call_regex)
                instance["result"] = [
                    {func_call.function.name: func_call.function.arguments}
                    for func_call in extracted_tool_response["tool_calls"]
                ]

                fout.write(json.dumps(instance) + "\n")

        shutil.move(temp_file, self.cfg.output_file)


GENERATION_TASK_CLASS = BFCLGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_bfcl_generation_config')
def bfcl_generation(cfg: BFCLGenerationConfig):
    cfg = BFCLGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = BFCLGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    BFCLGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        bfcl_generation()
