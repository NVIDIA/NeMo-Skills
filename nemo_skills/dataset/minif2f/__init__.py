# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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


# Default evaluation and generation settings for the minif2f dataset
PROMPT_CONFIG = 'lean4/formal-proof'
DATASET_GROUP = 'lean4'
METRICS_TYPE = "lean4-proof"
DEFAULT_EVAL_ARGS = "++eval_type=lean4"
DEFAULT_GENERATION_ARGS = ""
