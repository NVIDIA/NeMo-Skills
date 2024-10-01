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
from pathlib import Path

ANSWER_FIELD = "expected_answer"
ANSI = "ansi"
NAME_FOR_BASE_MODEL = "base_generation"
EXTRA_FIELDS = ["page_index", "file_name"]
CHAT_MODE = "chat_mode"
CHOOSE_MODEL = "choose generation"
CODE = "code"
CODE_SEPARATORS = {
    "code_begin": '<llm-code>',
    "code_end": '</llm-code>',
    "code_output_begin": '<llm-code-output>',
    "code_output_end": '</llm-code-output>',
}
CUSTOM = 'custom'
DATA_PAGE_SIZE = 10
EDIT_ICON_PATH = "assets/images/edit_icon.png"
SAVE_ICON_PATH = "assets/images/save_icon.png"
ERROR_MESSAGE_TEMPLATE = "When applying {} function\ngot errors\n{}"
FEW_SHOTS_INPUT = "few_shots_input"
FILE_NAME = 'file_name'
FILES_ONLY = "files_only"
FILES_FILTERING = "add_files_filtering"
GREEDY = "greedy"
IGNORE_FIELDS = ['stop_phrases', 'used_prompt', 'server_type']
QUESTIONS_FILTERING = "questions_filtering"
QUERY_INPUT_TYPE = "query_input"
QUERY_INPUT_ID = '{{"type": "{}", "id": "{}"}}'
QUESTION_FIELD = "problem"
ONE_SAMPLE_MODE = "one_sample"
METRICS = "metrics"
OUTPUT = "output"
OUTPUT_PATH = "{}-{}.jsonl"
PARAMS_TO_REMOVE = [
    'output_file',
    'dataset',
    'split',
    'example_dicts',
    'retriever',
    '_context_template',
    'save_generations_path',
]
PARAMETERS_FILE_NAME = "nemo_inspector/results/parameters.json"
TEMPLATES_FOLDER = os.path.join(Path(__file__).parents[2].absolute(), 'nemo_skills/prompt/template')
CONFIGS_FOLDER = os.path.join(Path(__file__).parents[2].absolute(), 'nemo_skills/prompt/config')
RETRIEVAL = 'retrieval'
RETRIEVAL_FIELDS = [
    'max_retrieved_chars_field',
    'retrieved_entries',
    'retrieval_file',
    'retrieval_field',
    'retrieved_few_shots',
    'max_retrieved_chars',
    'randomize_retrieved_entries',
]
STATS_KEYS = [
    'question_index',
    'problem',
]
SEPARATOR_DISPLAY = '.'
SEPARATOR_ID = '->'
SETTING_PARAMS = [
    'server',
    'sandbox',
    'output_file',
    'inspector_params',
    'types',
    'stop_phrases',
]
STATISTICS_FOR_WHOLE_DATASET = ["correct_answer", "wrong_answer", "no_answer"]
UNDEFINED = "undefined"
MARKDOWN = "markdown"
MODEL_SELECTOR_ID = '{{"type": "model_selector", "id": {}}}'
LABEL_SELECTOR_ID = '{{"type": "label_selector", "id": {}}}'
LABEL = "labels"
LATEX = "latex"
CHOOSE_LABEL = "choose label"
DELETE = "delete"
GENERAL_STATS = "general_stats"
