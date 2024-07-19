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
import urllib.request
from pathlib import Path

URL_QUESTIONS = "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl"
URL_BASELINE = (
    "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl"
)


if __name__ == "__main__":
    data_folder = Path(__file__).absolute().parent
    data_folder.mkdir(exist_ok=True)
    questions = str(data_folder / "question.jsonl")
    baseline = str(data_folder / "gpt-4-0314.jsonl")
    output_file = str(data_folder / "test.jsonl")
    urllib.request.urlretrieve(URL_QUESTIONS, questions)
    urllib.request.urlretrieve(URL_BASELINE, baseline)

    baseline_answers = {}
    with open(baseline, "rt", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            assert len(data['choices']) == 1
            assert len(data['choices'][0]['turns']) == 1
            baseline_answers[data['question_id']] = data['choices'][0]['turns'][0]['content']

    with open(questions, "rt", encoding="utf-8") as fin, open(output_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            assert len(data['turns']) == 1
            data['question'] = data.pop('turns')[0]['content']
            data['baseline_answer'] = baseline_answers[data['question_id']]
            # original code sets different temperature for different categories
            # TODO: do we need to do the same or greedy is good enough?

            fout.write(json.dumps(data) + "\n")
