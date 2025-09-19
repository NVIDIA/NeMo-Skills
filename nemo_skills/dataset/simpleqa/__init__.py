# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "math"
METRICS_TYPE = "simpleqa"
EVAL_ARGS = "++eval_type=math "
GENERATION_ARGS = "++prompt_config=generic/simpleqa "
EVAL_SPLIT = "test"

# SimpleQA requires judge model for evaluating factual accuracy
# Setting openai judge by default, but can be overridden from command line for a locally hosted model
# Using GPT-4 as recommended for factual evaluation tasks

JUDGE_PIPELINE_ARGS = {
    "model": "/hf_models/Qwen2.5-32B-Instruct",
    "server_type": "sglang",
    "server_gpus": 8,
    "server_nodes": 1,
    "server_args": "--context-length 8400",
}


JUDGE_ARGS = "++prompt_config=judge/simpleqa ++generation_key=judgement ++add_generation_stats=False"
