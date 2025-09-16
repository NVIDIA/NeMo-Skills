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
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_enhanced_model, server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class EnhancedGenerationConfig(GenerateSolutionsConfig):
    """Enhanced generation configuration with wrapper support.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.enhanced_generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Enable context passing to models (data_point and all_data parameters)
    enable_context_passing: bool = False

    # Model wrapper configuration (similar to tool_modules)
    #   List of wrapper provider locators using double-colon syntax for the wrapper class.
    #   Each item should be of the form:
    #     - Module class:  module.path.to.wrapper::ClassName
    #     - File class:    /abs/or/rel/path/to/wrapper.py::ClassName
    #
    #   Examples:
    #     - ++wrapper_modules=["nemo_skills.inference.wrappers.post_processing::PostProcessingWrapper"]
    #     - ++wrapper_modules=["/custom/path/eval_wrapper.py::EvalWrapper","nemo_skills.inference.wrappers.fixed_key::AddFixedKeyWrapper"]
    wrapper_modules: list[str] | None = None

    #   Per-wrapper overrides keyed by the Wrapper class name (the same ClassName used above).
    #   Use dotted keys to set nested values (e.g., sandbox.timeout).
    #
    #   Common patterns:
    #     - Set AddFixedKeyWrapper key/value:
    #         ++wrapper_overrides.AddFixedKeyWrapper.key="processed"
    #         ++wrapper_overrides.AddFixedKeyWrapper.value="true"
    wrapper_overrides: dict | None = field(default_factory=dict)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_enhanced_generation_config", node=EnhancedGenerationConfig)


class EnhancedGenerationTask(GenerationTask):
    """GenerationTask with optional wrapper support."""

    def __init__(self, cfg: EnhancedGenerationConfig):
        super().__init__(cfg)

    def setup_llm(self):
        # Set up sandbox as usual
        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None

        # Check if we need enhanced model with wrappers
        if self.cfg.wrapper_modules:
            # Prepare context to pass to wrappers
            wrapper_context = {
                "sandbox": self.sandbox,  # Pass the sandbox instance
                "cfg": self.cfg,  # Pass the full config
                "tokenizer": self.tokenizer,  # Pass tokenizer if needed
                "prompt": self.prompt,  # Pass prompt formatter
            }

            return get_enhanced_model(
                tokenizer=self.tokenizer,
                wrapper_modules=self.cfg.wrapper_modules,
                wrapper_overrides=self.cfg.wrapper_overrides,
                additional_config=wrapper_context,
                **self.cfg.server,
            )
        else:
            # Fall back to standard model creation logic from parent
            if self.cfg.code_execution:
                from nemo_skills.inference.model import get_code_execution_model

                return get_code_execution_model(**self.cfg.server, tokenizer=self.tokenizer, sandbox=self.sandbox)
            elif self.cfg.tool_modules is not None:
                from nemo_skills.inference.model import get_tool_calling_model

                return get_tool_calling_model(
                    **self.cfg.server,
                    tool_modules=self.cfg.tool_modules,
                    tool_overrides=self.cfg.tool_overrides,
                    tokenizer=self.tokenizer,
                    additional_config={"sandbox": self.cfg.sandbox},
                )
            else:
                from nemo_skills.inference.model import get_model

                return get_model(**self.cfg.server, tokenizer=self.tokenizer)

    async def process_single_datapoint(self, data_point, all_data):
        # Handle inference config - same as parent
        from dataclasses import asdict, is_dataclass

        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        generation_params = {
            **inference_params,
            **self.extra_generate_params,
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
        }

        # Only pass context if enabled
        if self.cfg.enable_context_passing:
            generation_params["data_point"] = data_point
            generation_params["all_data"] = all_data

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        return await self.llm.generate_async(**generation_params)


GENERATION_TASK_CLASS = EnhancedGenerationTask


# Update the hydra main to use the enhanced task
@hydra.main(version_base=None, config_name="base_enhanced_generation_config")
def generate(cfg: EnhancedGenerationConfig):
    cfg = EnhancedGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = EnhancedGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    EnhancedGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
