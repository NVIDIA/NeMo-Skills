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

import os

# running most things through subprocess since that's how it's usually used
import subprocess
from pathlib import Path

import pytest
from test_datasets import DATASETS

DATA_TO_TEST = []
template_folder = Path(__file__).parents[1] / 'nemo_skills' / 'prompt' / 'template'
prompt_templates = [f[:-5] for f in os.listdir(template_folder) if f.endswith('.yaml')]

for prompt_template in prompt_templates:
    for dataset, splits in DATASETS:
        for split in splits:
            DATA_TO_TEST.append((dataset, split, prompt_template))


@pytest.mark.parametrize("dataset,split,prompt_template", DATA_TO_TEST)
def test_generation_dryrun_default(dataset, split, prompt_template):
    """Testing the default prompts for each dataset."""
    cmd = (
        "python nemo_skills/inference/generate.py "
        f"    ++output_file=./test.jsonl "
        f"    ++prompt_template={prompt_template} "
        f"    ++dataset={dataset} "
        f"    ++split={split} "
        f"    ++server.server_type=nemo "
        f"    ++dry_run=True "
    )
    subprocess.run(cmd, shell=True, check=True)
