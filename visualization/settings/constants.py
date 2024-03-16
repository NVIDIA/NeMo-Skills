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

ANSWER_FIELD = "expected_answer"
EXTRA_FIELDS = ["page_index", "file_name"]
CHAT_MODE = "chat_mode"
CHOOSE_MODEL = "choose_generation"
DATA_PAGE_SIZE = 10
ERROR_MESSAGE_TEMPLATE = "When applying {} function\ngot errors\n{}"
FEW_SHOTS_INPUT = "few_shots_input"
GREEDY = "greedy"
QUERY_INPUT_TYPE = "query_input"
QUERY_INPUT_ID = '{{"type": "{}", "id": "{}"}}'
QUESTION_FIELD = "question"
ONE_SAMPLE_MODE = "one_sample"
METRICS = "metrics"
OUTPUT = "output"
OUTPUT_PATH = "{}-{}.jsonl"
PARAMS_FOR_WHOLE_DATASET_ONLY = ['offset', 'max_samples']
PARAMETERS_FILE_NAME = "visualization/results/parameters.json"
STATS_KEYS = [
    'question_index',
    'question',
]
SEPARATOR_DISPLAY = '.'
SEPARATOR_ID = '->'
UNDEFINED = "undefined"
WHOLE_DATASET_MODE = "whole_dataset"
MODEL_SELECTOR_ID = '{{"type": "model_selector", "id": {}}}'
LABEL_SELECTOR_ID = '{{"type": "label_selector", "id": {}}}'
LABEL = "labels"
CHOOSE_LABEL = "choose label"
DELETE = "delete"
GENERAL_STATS = "general_stats"
