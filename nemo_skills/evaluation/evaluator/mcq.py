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

import json
import logging
import re

from tqdm import tqdm

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def eval_mcq(cfg):
    def extract_letter(text):
        # extract prediction from boxed{}
        parsed = extract_answer(text)
        if parsed is not None and len(parsed) != 1:
            match = re.findall(r"\b[A-J]\b(?!.*\b[A-J]\b)", parsed, re.DOTALL)
            if len(match) > 0:
                parsed = match[-1].strip()

        # adapted from https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-overview
        if parsed is None:
            match = re.findall(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])", text)
            if match:
                parsed = match[-1].strip()
        return parsed

    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]
        with open(file, 'wt', encoding='utf-8') as fout:
            for sample in tqdm(data):
                sample['predicted_answer'] = extract_letter(sample["generation"])
                sample['symbolic_correct'] = sample['predicted_answer'] == sample['expected_answer']
                fout.write(json.dumps(sample) + "\n")
