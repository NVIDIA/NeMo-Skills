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

import json
import os
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/refs/heads/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
)

# corrected references are from https://github.com/lm-sys/FastChat/pull/3158
REF_URL = (
    "https://raw.githubusercontent.com/Zhilin123/FastChat/refs/heads/main"
    "/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4-0125-preview.jsonl"
)


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "original_test.json")
    original_file_ref = str(data_dir / "original_test_ref.json")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / "test.jsonl")

    if not os.path.exists(original_file):
        urllib.request.urlretrieve(URL, original_file)
    if not os.path.exists(original_file_ref):
        urllib.request.urlretrieve(REF_URL, original_file_ref)

    ref_data = {}

    with open(original_file_ref, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            entry = json.loads(line)
            ref_data[entry['question_id']] = entry

    data = []

    with open(original_file, "rt", encoding="utf-8") as fin:
        for index, line in enumerate(fin):
            entry = json.loads(line)
            turns = []
            for turn in entry["turns"]:
                turns.append({"question": turn})
            entry["turns"] = turns
            # adding reference answers to be used in llm-as-a-judge eval
            if entry['question_id'] in ref_data:
                ref_entry = ref_data[entry['question_id']]
                assert len(ref_entry['choices']) == 1
                assert len(ref_entry['choices'][0]['turns']) == 2

                entry['ref_answer_1'] = ref_entry['choices'][0]['turns'][0]
                entry['ref_answer_2'] = ref_entry['choices'][0]['turns'][1]

            data.append(entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")
