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

__version__ = '0.5.0'

_containers = {
    'trtllm': f'igitman/nemo-skills-trtllm:{__version__}',
    'vllm': f'igitman/nemo-skills-vllm:{__version__}',
    'sglang': f'igitman/nemo-skills-sglang:{__version__}',
    'nemo': f'igitman/nemo-skills-nemo:{__version__}',
    'sandbox': f'igitman/nemo-skills-sandbox:{__version__}',
    'nemo-skills': f'igitman/nemo-skills:{__version__}',
    'verl': f'igitman/nemo-skills-verl:{__version__}',
}
