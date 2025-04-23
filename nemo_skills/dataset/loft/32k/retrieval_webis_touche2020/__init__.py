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
# this will be copied and updated for each ruler subtask. 
# PROMPT_CONFIG = 'generic/default'
# DATASET_GROUP = "chat"
# METRICS_TYPE = "ruler"
# DEFAULT_EVAL_ARGS = "++eval_type=ruler ++eval_config.match_type=all"
# DEFAULT_GENERATION_ARGS = "++input_file=/loft/128k/arguana/test.jsonl ++dataset=null ++split=null"PROMPT_CONFIG = "generic/default"
DATASET_GROUP = "chat"
METRICS_TYPE = "loft"
DEFAULT_EVAL_ARGS = "++eval_type=loft ++eval_config.metrics=recall_at_k ++eval_config.k=1"
DEFAULT_GENERATION_ARGS = "++input_file=/loft/32k/retrieval_webis_touche2020/test.jsonl ++dataset=null ++split=null"
