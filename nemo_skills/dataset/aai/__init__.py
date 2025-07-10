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

IS_BENCHMARK_GROUP = True

SCORE_MODULE = "nemo_skills.dataset.aai.aai_score"

BENCHMARKS = {
    "mmlu-pro": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/mcq-10choices ++inference.temperature=0.0",
        # can add "NUM_CHUNKS": N to parallelize
    },
    "hle": {
        "GENERATION_ARGS": "++remove_thinking=True ++inference.temperature=0.0",
        "JUDGE_ARGS": "++prompt_config=judge/hle ++generation_key=judgement",
    },
    "gpqa": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/mcq-4choices ++inference.temperature=0.0",
    },
    "math-500": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/math ++inference.temperature=0.0",
        "NUM_SAMPLES": 3,
    },
    "aime24": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/math ++inference.temperature=0.0",
        "NUM_SAMPLES": 10,
    },
    "scicode": {
        "GENERATION_ARGS": "++inference.temperature=0.0",
        "NUM_SAMPLES": 3,
    },
    "livecodebench": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/livecodebench ++inference.temperature=0.0",
        "EVAL_SPLIT": "test_v5_2407_2412",
        "NUM_SAMPLES": 3,
    },
}
