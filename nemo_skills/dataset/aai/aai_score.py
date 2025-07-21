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


def compute_score(metrics: dict):
    mmlu_pro = metrics['mmlu-pro']['pass@1']['symbolic_correct']
    hle = metrics['hle']['pass@1']['judge_correct']
    gpqa = metrics['gpqa']['pass@1']['symbolic_correct']

    aime24 = metrics['aime24']['pass@1[avg-of-10]']['symbolic_correct']
    math500 = metrics['math-500']['pass@1[avg-of-3]']['symbolic_correct']

    scicode = metrics['scicode']['pass@1[avg-of-3]']['subtask_accuracy']
    livecodebench = metrics['livecodebench']['pass@1[avg-of-3]']['accuracy']

    math_score = (aime24 + math500) / 2
    code_score = (scicode + livecodebench) / 2
    overall_score = (mmlu_pro + hle + gpqa) / 6 + (math_score + code_score) / 4
    return {
        'overall_score': overall_score,
        'math_score': math_score,
        'code_score': code_score,
        'mmlu_pro': mmlu_pro,
        'hle': hle,
        'gpqa': gpqa,
        'aime24': aime24,
        'math500': math500,
        'scicode': scicode,
        'livecodebench': livecodebench,
    }
