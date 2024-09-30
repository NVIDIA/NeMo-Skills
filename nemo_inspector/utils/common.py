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

import datetime
import functools
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import fields, is_dataclass
from types import NoneType
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, get_args, get_origin, get_type_hints

from dash import html
from flask import current_app
from joblib import Parallel, delayed
from settings.constants import (
    ANSWER_FIELD,
    ERROR_MESSAGE_TEMPLATE,
    FILE_NAME,
    OUTPUT,
    PARAMETERS_FILE_NAME,
    PARAMS_TO_REMOVE,
    QUESTION_FIELD,
    RETRIEVAL_FIELDS,
    SEPARATOR_DISPLAY,
    SETTING_PARAMS,
    STATS_KEYS,
    UNDEFINED,
)

from nemo_skills.inference.generate import GenerateSolutionsConfig, InferenceConfig
from nemo_skills.prompt.utils import FewShotExamplesConfig, PromptConfig, PromptTemplate
from nemo_skills.utils import unroll_files

custom_stats = {}
general_custom_stats = {}
deleted_stats = set()
excluded_rows = set()
editable_rows = set()


def get_editable_rows() -> Set:
    return editable_rows


def get_excluded_row() -> Set:
    return excluded_rows


def get_deleted_stats() -> Set:
    return deleted_stats


def get_custom_stats() -> Dict:
    return custom_stats


def get_general_custom_stats() -> Dict:
    return general_custom_stats


def parse_model_answer(answer: str) -> List[Dict]:
    """
    Parses a model answer and extracts code blocks, explanations, and outputs preserving their sequence.

    Args:
        answer (str): The model answer to parse.

    Returns:
        List[Dict]: A list of dictionaries containing the parsed results. Each dictionary
        contains the following keys:
            - 'explanation': The explanation text before the code block.
            - 'code': The code block.
            - 'output': The output of the code block.

    """
    config = current_app.config['nemo_inspector']
    code_start, code_end = map(
        re.escape,
        config["inspector_params"]["code_separators"],
    )
    output_start, output_end = map(
        re.escape,
        config["inspector_params"]["code_output_separators"],
    )
    code_pattern = re.compile(fr'{code_start}(.*?){code_end}', re.DOTALL)
    code_output_pattern = re.compile(
        fr'{code_start}((?:(?!{code_end}).)*){code_end}\s*\n\s*{output_start}((?:(?!{output_end}).)*){output_end}',
        re.DOTALL,
    )
    code_matches = list(code_pattern.finditer(answer))
    code_output_matches = list(code_output_pattern.finditer(answer))
    parsed_results = []
    last_index = 0
    for code_match in code_matches:
        explanation = answer[last_index : code_match.start()].strip()
        code_text = code_match.group(1).strip()
        output_text = None
        if code_output_matches and code_output_matches[0].start() == code_match.start():
            output_match = code_output_matches.pop(0)
            output_text = output_match.group(2).strip()
        parsed_results.append(
            {
                'explanation': explanation,
                'code': code_text,
                'output': output_text,
            }
        )
        last_index = code_match.end()
        if output_text is not None:
            last_index = output_match.end()
    if last_index < len(answer):
        trailing_text = answer[last_index:].strip()
        if code_start.replace("\\", "") in trailing_text:
            code_start_index = trailing_text.find(code_start.replace("\\", ""))
            parsed_results.append(
                {
                    'explanation': trailing_text[0:code_start_index].strip(),
                    'code': trailing_text[code_start_index + len(code_start.replace("\\", "")) :],
                    'output': "code_block was not finished",
                    'wrong_code_block': True,
                }
            )
            trailing_text = None
        if trailing_text:
            parsed_results.append({'explanation': trailing_text, 'code': None, 'output': None})
    return parsed_results


def get_height_adjustment() -> html.Iframe:
    return html.Iframe(
        id="query_params_iframe",
        srcDoc="""
        <!DOCTYPE html>
        <html>
        <head>
        </head>
        <body>
            <script>
                window.addEventListener('DOMContentLoaded', function() {
                    parent.registerTextarea();
                });
            </script>
        </body>
        </html>
        """,
        style={"visibility": "hidden"},
    )


@functools.lru_cache()
def get_test_data(index: int, dataset: str) -> Tuple[Dict, int]:
    if not dataset or dataset == UNDEFINED or os.path.isfile(dataset) is False:
        return {QUESTION_FIELD: "", ANSWER_FIELD: ""}, 0
    with open(dataset) as file:
        tests = file.readlines()
        index = max(min(len(tests), index), 1)
        test = json.loads(tests[index - 1])
    return test, index


def get_values_from_input_group(children: Iterable) -> Dict:
    values = {}
    for child in children:
        for input_group_child in child["props"]["children"]:
            if "id" in input_group_child["props"].keys() and "value" in input_group_child["props"].keys():
                type_function = str
                value = input_group_child["props"]["value"]
                id = (
                    input_group_child["props"]["id"]["id"]
                    if isinstance(input_group_child["props"]["id"], Dict)
                    else input_group_child["props"]["id"]
                )
                if value is None or value == UNDEFINED:
                    values[id] = None
                    continue
                if str(value).isdigit() or str(value).replace("-", "", 1).isdigit():
                    type_function = int
                elif str(value).replace(".", "", 1).replace("-", "", 1).isdigit():
                    type_function = float

                values[id] = type_function(str(value).replace('\\n', '\n'))

    return values


def extract_query_params(query_params_ids: List[Dict], query_params: List[Dict]) -> Dict:
    default_answer = {"question": "", "expected_answer": ""}
    try:
        query_params_extracted = {param_id['id']: param for param_id, param in zip(query_params_ids, query_params)}
    except ValueError:
        query_params_extracted = default_answer

    return query_params_extracted or default_answer


def get_utils_from_config_helper(cfg: Dict, display_path: bool = True) -> Dict:
    config = {}
    for key, value in sorted(cfg.items()):
        if key in PARAMS_TO_REMOVE:
            continue
        elif isinstance(value, Dict):
            config = {
                **config,
                **{
                    (key + SEPARATOR_DISPLAY if display_path and 'template' in inner_key else "") + inner_key: value
                    for inner_key, value in get_utils_from_config_helper(value).items()
                },
            }
        elif not isinstance(value, List):
            config[key] = value
    return config


def get_utils_from_config(cfg: Dict, display_path: bool = True) -> Dict:
    return {
        SEPARATOR_DISPLAY.join(key.split(SEPARATOR_DISPLAY)[1:]) or key: value
        for key, value in get_utils_from_config_helper(cfg, display_path).items()
        if key not in RETRIEVAL_FIELDS
    }


def get_stats(all_files_data: List[Dict]) -> Tuple[float, float, float]:
    """Returns the percentage of correct, wrong, and no response answers in the given data.

    If not data is provided, returns -1 for all values.
    """
    correct = 0
    wrong = 0
    no_response = 0
    for data in all_files_data:
        if data.get("predicted_answer") is None:
            no_response += 1
        elif data.get("is_correct", False):
            correct += 1
        else:
            wrong += 1

    if len(all_files_data):
        return (
            correct / len(all_files_data),
            wrong / len(all_files_data),
            no_response / len(all_files_data),
        )
    return -1, -1, -1


def get_metrics(all_files_data: List[Dict], errors_dict: Dict = {}) -> Dict:
    correct_responses, wrong_responses, no_response = get_stats(all_files_data)
    custom_stats = {}
    for name, func in get_custom_stats().items():
        if name not in errors_dict:
            errors_dict[name] = {}
        custom_stats[name] = catch_eval_exception(
            [],
            func,
            all_files_data,
            "Got error when applying function",
            errors_dict[name],
        )

    stats = {
        'correct_responses': round(correct_responses, 2),
        "wrong_responses": round(wrong_responses, 2),
        "no_response": round(no_response, 2),
        **custom_stats,
    }
    return stats


def get_eval_function(text):
    template = """
def eval_function(data):
{}
    return {}
"""
    code_lines = [''] + text.strip().split('\n')
    code = template.format(
        '\n    '.join(code_lines[:-1]),
        code_lines[-1:][0],
    )
    namespace = {}
    exec(code, namespace)
    return namespace['eval_function']


def calculate_metrics_for_whole_data(table_data: List, model_id: str) -> Dict:
    errors_dict = {}
    for question_id in range(len(table_data)):
        stats = get_metrics(table_data[question_id][model_id], errors_dict)
        table_data[question_id][model_id] = list(
            map(
                lambda data: {**data, **stats},
                table_data[question_id][model_id],
            )
        )
    if len(errors_dict):
        for name, error_dict in errors_dict.items():
            logging.error(ERROR_MESSAGE_TEMPLATE.format(name, error_dict))


def catch_eval_exception(
    available_models: List[str],
    eval_func: Callable[[Dict], bool],
    data: Dict,
    default_answer: Union[bool, str],
    errors_dict: Optional[Dict] = {},
) -> bool:
    try:
        if eval_func is None:
            return default_answer
        return eval_func(data)
    except Exception as e:
        if str(e).split(" ")[-1].replace("'", "") not in available_models:
            if str(e) not in errors_dict:
                errors_dict[str(e)] = 0
            errors_dict[str(e)] += 1
        return default_answer


def custom_deepcopy(data) -> List:
    new_data = []
    for item in data:
        new_item = {}
        for key, value_list in item.items():
            new_item[key] = value_list
        new_data.append(new_item)
    return new_data


@functools.lru_cache(maxsize=1)
def get_data_from_files(cache_indicator=None) -> List:
    if cache_indicator is not None:
        return []
    base_config = current_app.config['nemo_inspector']
    dataset = None
    if base_config['data_file'] != UNDEFINED:
        with open(base_config['data_file']) as f:
            dataset = [json.loads(line) for line in f]

    available_models = {
        model_name: model_info["file_paths"] for model_name, model_info in get_available_models().items()
    }

    all_models_data_array = []

    def process_model_files(model_id, results_files, dataset):
        model_data = defaultdict(list)
        file_names = {}
        for file_id, path in enumerate(results_files):
            file_name = path.split('/')[-1].split('.')[0]
            if file_name in file_names:
                file_names[file_name] += 1
                file_name += f"_{file_names[file_name]}"
            else:
                file_names[file_name] = 1
            with open(path) as f:
                answers = map(json.loads, f)
                for question_index, answer in enumerate(answers):
                    result = {
                        FILE_NAME: file_name,
                        **(dataset[question_index] if dataset and len(dataset) > question_index else {}),
                        "question_index": question_index + 1,
                        "page_index": file_id,
                        "labels": [],
                        **answer,
                    }
                    model_data[question_index].append(result)
        return model_id, model_data

    num_cores = -1
    model_data_list = Parallel(n_jobs=num_cores)(
        delayed(process_model_files)(model_id, results_files, dataset)
        for model_id, results_files in available_models.items()
    )

    for model_id, model_data in model_data_list:
        for question_index, results in model_data.items():
            if len(all_models_data_array) <= question_index:
                all_models_data_array.append({})
            all_models_data_array[question_index][model_id] = results
            stats = get_metrics(all_models_data_array[question_index][model_id])
            all_models_data_array[question_index][model_id] = list(
                map(
                    lambda data: {**data, **stats},
                    all_models_data_array[question_index][model_id],
                )
            )

    return all_models_data_array


def get_filtered_files(
    filter_function: str,
    sorting_function: str,
    array_to_filter: List,
) -> List:
    filter_lambda_functions = [
        get_eval_function(func.strip()) for func in (filter_function if filter_function else "True").split('&&')
    ]
    available_models = get_available_models()
    filtered_data = [
        list(
            filter(
                lambda data: catch_eval_exception(available_models, function, data, False),
                array_to_filter,
            )
        )
        for function in filter_lambda_functions
    ]

    filtered_data = list(filter(lambda data: data != [], filtered_data))
    filtered_data = filtered_data[0] if len(filtered_data) > 0 else [{FILE_NAME: ""}]
    if sorting_function and filtered_data != [{FILE_NAME: ""}]:
        sorting_lambda_function = get_eval_function(sorting_function.strip())
        filtered_data.sort(key=lambda data: catch_eval_exception(available_models, sorting_lambda_function, data, 0))

    return filtered_data


def is_detailed_answers_rows_key(key: str) -> bool:
    return (
        key not in get_deleted_stats()
        and 'index' not in key
        and key not in STATS_KEYS + list(get_metrics([]).keys())
        or key == 'question'
    )


@functools.lru_cache(maxsize=1)
def get_available_models(cache_indicator=None) -> Dict:
    if cache_indicator is not None:
        return {}
    try:
        with open(PARAMETERS_FILE_NAME) as f:
            runs_storage = json.load(f)
    except FileNotFoundError:
        runs_storage = {}
    models = list(runs_storage.keys())
    config = current_app.config["nemo_inspector"]["inspector_params"]
    for model_name in models:
        runs_storage[model_name]["file_paths"] = list(
            unroll_files([os.path.join(config["results_path"], model_name, f"{OUTPUT}*.jsonl")])
        )
    for model_name, files in config["model_prediction"].items():
        runs_storage[model_name] = {
            "utils": {},
            "examples": {},
            "file_paths": files,
        }

    return runs_storage


def run_subprocess(command: str) -> Tuple[str, bool]:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    success = True

    delta = datetime.timedelta(minutes=1)
    start_time = datetime.datetime.now()
    while result.returncode != 0 and datetime.datetime.now() - start_time <= delta:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logging.info(f"Error while running command: {command}")
        logging.info(f"Return code: {result.returncode}")
        logging.info(f"Output (stderr): {result.stderr.strip()}")
        success = False

    return result.stdout.strip(), result.stderr.strip(), success


def get_config(
    config_class: Union[GenerateSolutionsConfig, PromptConfig, InferenceConfig, FewShotExamplesConfig],
    utils: Dict[str, str],
    config: Dict,
) -> Union[GenerateSolutionsConfig, PromptConfig, InferenceConfig, FewShotExamplesConfig]:
    return config_class(
        **{
            key: value
            for key, value in {
                **config,
                **utils,
            }.items()
            if key in {field.name for field in fields(config_class)}
        },
    )


@functools.lru_cache(maxsize=1)
def get_settings():
    def get_settings_helper(config: Dict):
        settings = {}
        for key, value in config.items():
            if key in SETTING_PARAMS:
                settings[key] = value
            if isinstance(value, dict):
                settings = {**settings, **get_settings_helper(value)}
        return settings

    return get_settings_helper(current_app.config['nemo_inspector'])


def get_utils_dict(name: Union[str, Dict], value: Union[str, int], id: Union[str, Dict] = None):
    if id is None:
        id = name
    if name in current_app.config['nemo_inspector']['types'].keys():
        template = {
            'props': {
                'id': id,
                'options': [
                    {"label": value, "value": value} for value in current_app.config['nemo_inspector']['types'][name]
                ],
                'value': current_app.config['nemo_inspector']['types'][name][0],
            },
            'type': 'Select',
            'namespace': 'dash_bootstrap_components',
        }
    elif isinstance(value, (int, float)):
        float_params = {"step": 0.1} if isinstance(value, float) else {}
        template = {
            'props': {
                'id': id,
                'debounce': True,
                'min': 0,
                'type': 'number',
                'value': value,
                **float_params,
            },
            'type': 'Input',
            'namespace': 'dash_bootstrap_components',
        }
    else:
        template = {
            'props': {
                'id': id,
                'debounce': True,
                'style': {'width': '100%'},
                'value': value,
            },
            'type': 'Textarea',
            'namespace': 'dash_bootstrap_components',
        }
    return {
        'props': {
            'children': [
                {
                    'props': {'children': name},
                    'type': 'InputGroupText',
                    'namespace': 'dash_bootstrap_components',
                },
                template,
            ],
            'className': 'mb-3',
        },
        'type': 'InputGroup',
        'namespace': 'dash_bootstrap_components',
    }


def initialize_default(
    cls: Union[PromptTemplate, PromptConfig], specification={}
) -> Union[PromptTemplate, PromptConfig]:
    def get_default(field, specification={}):
        _type = get_type_hints(cls)[field.name]
        if field.name in specification:
            return specification[field.name]
        if is_dataclass(_type):
            return initialize_default(_type, specification)
        else:
            args = get_args(_type)
            if len(args):
                if NoneType in args:
                    return None
                else:
                    return args[0]()
            return (get_origin(_type) or _type)()

    return cls(**{field.name: get_default(field, specification) for field in fields(cls)})
