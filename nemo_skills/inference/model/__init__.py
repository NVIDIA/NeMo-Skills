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

from nemo_skills.utils import python_doc_to_cmd_help

from .azure import AzureOpenAIModel

# Base classes
from .base import BaseModel, BaseRewardModel, OpenAIAPIModel

# Code execution
from .code_execution import CodeExecutionConfig, CodeExecutionWrapper
from .megatron import MegatronModel
from .nemo import NemoModel, NemoRewardModel
from .openai import OpenAIModel

# Model implementations
from .trtllm import TRTLLMModel

# Utilities
from .utils import RequestException, trim_after_stop_phrases
from .vllm import VLLMModel, VLLMRewardModel

# Model registry
models = {
    'trtllm': TRTLLMModel,
    'trtllm-serve': VLLMModel,
    'nemo': NemoModel,
    'megatron': MegatronModel,
    'openai': OpenAIModel,
    'azureopenai': AzureOpenAIModel,
    'vllm': VLLMModel,
    'sglang': VLLMModel,
}

# Reward model registry
reward_models = {
    'nemo': NemoRewardModel,
    'vllm': VLLMRewardModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)


def get_reward_model(server_type, model_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = reward_models[server_type.lower()]
    return model_class(model_type=model_type, **kwargs)


def get_code_execution_model(server_type, code_execution=None, sandbox=None, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model = get_model(server_type=server_type, **kwargs)
    if code_execution is None:
        code_execution = {}
    code_execution_config = CodeExecutionConfig(**code_execution)
    return CodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)


def server_params():
    """Returns server documentation (to include in cmd help)."""
    # TODO: This needs a fix now
    prefix = f'\n        server_type: str = MISSING - Choices: {list(models.keys())}'
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")
