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

__version__ = '0.6.0'

# only used in ns setup command to initialize with defaults
_containers = {
    'trtllm': 'igitman/nemo-skills-trtllm:0.6.0',
    'vllm': 'igitman/nemo-skills-vllm:0.6.0',
    'sglang': 'igitman/nemo-skills-sglang:0.6.0',
    'nemo': 'igitman/nemo-skills-nemo:0.6.0',
    'sandbox': 'igitman/nemo-skills-sandbox:0.6.0',
    'nemo-skills': 'igitman/nemo-skills:0.6.0',
    'verl': 'igitman/nemo-skills-verl:0.6.0',
}
