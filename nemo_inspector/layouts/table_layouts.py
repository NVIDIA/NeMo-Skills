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
import logging
import math
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import dash_table, html
from layouts.base_layouts import (
    get_code_text_area_layout,
    get_selector_layout,
    get_single_prompt_output_layout,
    get_switch_layout,
    get_text_modes_layout,
)
from settings.constants import (
    ANSI,
    CODE,
    COMPARE,
    COMPARE_ICON_PATH,
    DATA_PAGE_SIZE,
    EDIT_ICON_PATH,
    ERROR_MESSAGE_TEMPLATE,
    FILE_NAME,
    FILES_FILTERING,
    FILES_ONLY,
    LABEL,
    LATEX,
    MODEL_SELECTOR_ID,
    NAME_FOR_BASE_MODEL,
    QUESTIONS_FILTERING,
    STATS_KEYS,
)
from utils.common import (
    catch_eval_exception,
    custom_deepcopy,
    get_available_models,
    get_compared_rows,
    get_data_from_files,
    get_editable_rows,
    get_eval_function,
    get_excluded_row,
    get_filtered_files,
    get_general_custom_stats,
    get_metrics,
    is_detailed_answers_rows_key,
)

table_data = []
labels = []


def get_table_data() -> List:
    return table_data


def get_labels() -> List:
    return labels


def get_filter_text(available_filters: List[str] = [], mode: str = FILES_FILTERING) -> str:
    available_filters = list(
        get_table_data()[0][list(get_table_data()[0].keys())[0]][0].keys()
        if len(get_table_data()) and not available_filters
        else STATS_KEYS + list(get_metrics([]).keys()) + ["+ all fields in json"]
    )
    if mode == FILES_ONLY:
        return (
            "Write an expression to filter the data\n\n"
            + "For example:\ndata['is_correct'] and not data['error_message']\n\n"
            + "The expression has to return bool.\n\n"
            + "Available parameters to filter data:\n"
            + '\n'.join(
                [', '.join(available_filters[start : start + 5]) for start in range(0, len(available_filters), 5)]
            ),
        )
    elif mode == FILES_FILTERING:
        return (
            "Write an expression to filter the data\n"
            + "Separate expressions for different generations with &&\n"
            + "You can use base_generation variable to access data from the current generation\n\n"
            + "For example:\ndata['generation1']['correct_responses'] > 0.5 && data[base_generation]['no_response'] < 0.2\n\n"
            + "The expression has to return bool.\n\n"
            + "Available parameters to filter data:\n"
            + '\n'.join(
                [', '.join(available_filters[start : start + 5]) for start in range(0, len(available_filters), 5)]
            ),
        )
    elif mode == QUESTIONS_FILTERING:
        return (
            "Write an expression to filter the data\n"
            + "You can operate with a dictionary containing keys representing generation names\n"
            + "and a list of values as JSON data from your generation from each file.\n"
            + "You can use base_generation variable to access data from the current generation\n\n"
            + "For example:\ndata['generation1'][0]['is_correct'] != data[base_generation][0]['is_correct']\n\n"
            + "The expression has to return bool.\n\n"
            + "Available parameters to filter data:\n"
            + '\n'.join(
                [', '.join(available_filters[start : start + 5]) for start in range(0, len(available_filters), 5)]
            ),
        )


def get_filter_layout(id: int = -1, available_filters: List[str] = [], mode: str = FILES_FILTERING) -> html.Div:
    text = get_filter_text(available_filters, mode)

    filter_mode = (
        [
            get_switch_layout(
                id={"type": "filter_mode", "id": id},
                labels=["filter files"],
                is_active=True,
                additional_params={
                    "inline": True,
                    "style": {"margin-left": "10px"},
                },
            )
        ]
        if mode != FILES_ONLY
        else []
    )

    header = dbc.ModalHeader(
        (
            [
                dbc.ModalTitle(
                    "Set Up Your Filter",
                ),
            ]
            + filter_mode
        ),
        close_button=True,
    )
    body = dbc.ModalBody(
        html.Div(
            [
                html.Pre(text, id={"type": "filter_text", "id": id}),
                get_code_text_area_layout(
                    id={
                        "type": "filter_function_input",
                        "id": id,
                    },
                ),
            ]
        )
    )
    switch = get_switch_layout(
        {
            "type": "apply_on_filtered_data",
            "id": id,
        },
        ["Apply for filtered data"],
        additional_params={"style": {"margin-left": "10px"}},
    )
    footer = dbc.ModalFooter(
        dbc.Button(
            "Apply",
            id={"type": "apply_filter_button", "id": id},
            className="ms-auto",
            n_clicks=0,
        )
    )
    return html.Div(
        [
            dbc.Button(
                "Filters",
                id={"type": "set_filter_button", "id": id},
                class_name='button-class',
            ),
            dbc.Modal(
                [
                    header,
                    body,
                    switch,
                    footer,
                ],
                size="lg",
                id={"type": "filter", "id": id},
                centered=True,
                is_open=False,
            ),
        ],
        style={'display': 'inline-block'},
    )


def get_sorting_layout(id: int = -1, available_params: List[str] = []) -> html.Div:
    available_params = list(
        get_table_data()[0][list(get_table_data()[0].keys())[0]][0].keys()
        if len(get_table_data()) and not available_params
        else STATS_KEYS + list(get_metrics([]).keys()) + ["+ all fields in json"]
    )
    text = (
        "Write an expression to sort the data\n\n"
        "For example: len(data['question'])\n\n"
        "The function has to return sortable type\n\n"
        "Available parameters to sort data:\n"
        + '\n'.join([', '.join(available_params[start : start + 5]) for start in range(0, len(available_params), 5)])
    )
    header = dbc.ModalHeader(
        dbc.ModalTitle("Set Up Your Sorting Parameters"),
        close_button=True,
    )
    body = dbc.ModalBody(
        html.Div(
            [
                html.Pre(text),
                get_code_text_area_layout(
                    id={
                        "type": "sorting_function_input",
                        "id": id,
                    },
                ),
            ],
        )
    )
    footer = dbc.ModalFooter(
        dbc.Button(
            "Apply",
            id={"type": "apply_sorting_button", "id": id},
            className="ms-auto",
            n_clicks=0,
        )
    )
    return html.Div(
        [
            dbc.Button(
                "Sort",
                id={"type": "set_sorting_button", "id": id},
                class_name='button-class',
            ),
            dbc.Modal(
                [
                    header,
                    body,
                    footer,
                ],
                size="lg",
                id={"type": "sorting", "id": id},
                centered=True,
                is_open=False,
            ),
        ],
        style={'display': 'inline-block'},
    )


def get_change_label_layout(id: int = -1, apply_for_all_files: bool = True) -> html.Div:
    header = dbc.ModalHeader(
        dbc.ModalTitle("Manage labels"),
        close_button=True,
    )
    switch_layout = (
        [
            get_switch_layout(
                {
                    "type": "aplly_for_all_files",
                    "id": id,
                },
                ["Apply for all files"],
                additional_params={"style": {"margin-left": "10px"}},
            )
        ]
        if apply_for_all_files
        else []
    )
    body = dbc.ModalBody(
        html.Div(
            [
                get_selector_layout(
                    options=labels,
                    id={"type": "label_selector", "id": id},
                    value="choose label",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id={
                                "type": "new_label_input",
                                "id": id,
                            },
                            placeholder="Enter new label",
                            type="text",
                        ),
                        dbc.Button(
                            "Add",
                            id={
                                "type": "add_new_label_button",
                                "id": id,
                            },
                        ),
                    ]
                ),
                *switch_layout,
                html.Pre("", id={"type": "chosen_label", "id": id}),
            ],
        )
    )
    footer = dbc.ModalFooter(
        html.Div(
            [
                dbc.Button(
                    children="Delete",
                    id={
                        "type": "delete_label_button",
                        "id": id,
                    },
                    className="ms-auto",
                    n_clicks=0,
                ),
                html.Pre(
                    " ",
                    style={'display': 'inline-block', 'font-size': '5px'},
                ),
                dbc.Button(
                    children="Apply",
                    id={"type": "apply_label_button", "id": id},
                    className="ms-auto",
                    n_clicks=0,
                ),
            ],
        ),
        style={'display': 'inline-block'},
    )
    return html.Div(
        [
            dbc.Button(
                "Labels",
                id={"type": "set_file_label_button", "id": id},
                class_name='button-class',
            ),
            dbc.Modal(
                [header, body, footer],
                size="lg",
                id={"type": "label", "id": id},
                centered=True,
                is_open=False,
            ),
        ],
        style={'display': 'inline-block'},
    )


def get_stats_layout() -> List[dbc.Row]:
    return [
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id='datatable',
                    columns=[
                        {
                            'name': name,
                            'id': name,
                            'hideable': True,
                        }
                        for name in STATS_KEYS + list(get_metrics([]).keys())
                    ],
                    row_selectable='single',
                    cell_selectable=False,
                    page_action='custom',
                    page_current=0,
                    page_size=DATA_PAGE_SIZE,
                    page_count=math.ceil(len(table_data) / DATA_PAGE_SIZE),
                    style_cell={
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 0,
                        'textAlign': 'center',
                    },
                    style_header={
                        'color': 'text-primary',
                        'text_align': 'center',
                        'height': 'auto',
                        'whiteSpace': 'normal',
                    },
                    css=[
                        {
                            'selector': '.dash-spreadsheet-menu',
                            'rule': 'position:absolute; bottom: 8px',
                        },
                        {
                            'selector': '.dash-filter--case',
                            'rule': 'display: none',
                        },
                        {
                            'selector': '.column-header--hide',
                            'rule': 'display: none',
                        },
                    ],
                ),
            )
        ),
    ]


def get_models_selector_table_cell(models: List[str], name: str, id: int, add_del_button: bool = False) -> dbc.Col:
    del_model_layout = (
        [
            dbc.Button(
                "-",
                id={"type": "del_model", "id": id},
                outline=True,
                color="primary",
                className="me-1",
                style={"height": "40px"},
            ),
        ]
        if add_del_button
        else []
    )
    return dbc.Col(
        html.Div(
            [
                html.Div(
                    get_selector_layout(
                        models,
                        json.loads(MODEL_SELECTOR_ID.format(id)),
                        name,
                    ),
                ),
                get_sorting_layout(id),
                get_filter_layout(id, mode=FILES_ONLY),
                get_change_label_layout(id),
            ]
            + del_model_layout
            + [get_text_modes_layout(id)],
            style={'display': 'inline-flex'},
        ),
        class_name='mt-1 bg-light font-monospace text-break small rounded border',
        id={"type": "column_header", "id": id},
    )


def get_models_selector_table_header(models: List[str]) -> List[dbc.Row]:
    return [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        "",
                    ),
                    width=2,
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                    id='first_column',
                )
            ]
            + [
                get_models_selector_table_cell(get_available_models(), name, i, i != 0)
                for i, name in enumerate(models)
            ],
            id='detailed_answers_header',
        )
    ]


def get_detailed_answer_column(id: int, file_id=None) -> dbc.Col:
    return dbc.Col(
        html.Div(
            children=(
                get_selector_layout([], {"type": "file_selector", "id": file_id}, "") if file_id is not None else ""
            ),
            id={
                'type': 'detailed_models_answers',
                'id': id,
            },
        ),
        class_name='mt-1 bg-light font-monospace text-break small rounded border',
    )


def get_detailed_answers_rows(keys: List[str], colums_number: int) -> List[dbc.Row]:
    return [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        html.Div(
                            [
                                html.Div(
                                    key,
                                    id={"type": "row_name", "id": i},
                                    style={"display": "inline-block"},
                                ),
                                dbc.Button(
                                    html.Img(
                                        src=EDIT_ICON_PATH,
                                        id={"type": "edit_row_image", "id": i},
                                        style={
                                            "height": "15px",
                                            "display": "inline-block",
                                        },
                                    ),
                                    id={"type": "edit_row_button", "id": i},
                                    outline=True,
                                    color="primary",
                                    className="me-1",
                                    style={
                                        "border": "none",
                                        "line-height": "1.2",
                                        "display": "inline-block",
                                        "margin-left": "1px",
                                        "display": "none" if key in (FILE_NAME, LABEL) else "inline-block",
                                    },
                                ),
                                dbc.Button(
                                    html.Img(
                                        src=COMPARE_ICON_PATH,
                                        id={"type": "compare_texts", "id": i},
                                        style={
                                            "height": "15px",
                                            "display": "inline-block",
                                        },
                                    ),
                                    id={"type": "compare_texts_button", "id": i},
                                    outline=True,
                                    color="primary",
                                    className="me-1",
                                    style={
                                        "border": "none",
                                        "line-height": "1.2",
                                        "display": "inline-block",
                                        "margin-left": "-10px" if key != LABEL else "1px",
                                        "display": "none" if key == FILE_NAME else "inline-block",
                                    },
                                ),
                                dbc.Button(
                                    "-",
                                    id={"type": "del_row", "id": i},
                                    outline=True,
                                    color="primary",
                                    className="me-1",
                                    style={
                                        "border": "none",
                                        "display": "inline-block",
                                        "margin-left": "-9px" if key != FILE_NAME else "1px",
                                    },
                                ),
                            ],
                            style={"display": "inline-block"},
                        ),
                    ),
                    width=2,
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                )
            ]
            + [get_detailed_answer_column(j * len(keys) + i) for j in range(colums_number)],
            id={"type": "detailed_answers_row", "id": i},
        )
        for i, key in enumerate(keys)
    ]


def get_table_answers_detailed_data_layout(
    models: List[str],
    keys: List[str],
) -> List[dbc.Row]:
    return get_models_selector_table_header(models) + get_detailed_answers_rows(keys, len(models))


def get_row_detailed_inner_data(
    question_id: int,
    model: str,
    rows_names: List[str],
    files_names: List[str],
    file_id: int,
    col_id: int,
    compare_to: Dict = {},
    text_modes: List[str] = [CODE, LATEX, ANSI],
) -> List:
    table_data = get_table_data()[question_id].get(model, [])
    row_data = []
    empty_list = False
    if table_data[file_id].get(FILE_NAME, None) not in files_names:
        empty_list = True
    for key in filter(
        lambda key: is_detailed_answers_rows_key(key),
        rows_names,
    ):
        if file_id < 0 or len(table_data) <= file_id or key in get_excluded_row():
            value = ""
        elif key == FILE_NAME:
            value = get_selector_layout(
                files_names,
                {"type": "file_selector", "id": col_id},
                (table_data[file_id].get(key, None) if not empty_list else ""),
            )
        elif empty_list:
            value = ""
        elif key in get_editable_rows():
            value = str(table_data[file_id].get(key, None))
        else:
            value = get_single_prompt_output_layout(
                str(table_data[file_id].get(key, None)),
                text_modes + ([COMPARE] if key in get_compared_rows() else []),
                str(compare_to.get(key, "")),
            )
        row_data.append(
            value
            if key not in get_editable_rows()
            else dbc.Textarea(id={"type": "editable_row", "id": key, "model_name": model}, value=value)
        )
    return row_data


def get_table_detailed_inner_data(
    question_id: int,
    rows_names: List[str],
    models: List[str],
    files_id: List[int],
    filter_functions: List[str],
    sorting_functions: List[str],
    text_modes: List[List[str]],
) -> List:
    table_data = []
    for col_id, (model, file_id, filter_function, sorting_function, modes) in enumerate(
        zip(models, files_id, filter_functions, sorting_functions, text_modes)
    ):
        row_data = get_row_detailed_inner_data(
            question_id=question_id,
            model=model,
            rows_names=rows_names,
            files_names=[
                file[FILE_NAME]
                for file in get_filtered_files(
                    filter_function,
                    sorting_function,
                    get_table_data()[question_id][model] if len(get_table_data()) else [],
                )
            ],
            file_id=file_id,
            col_id=col_id,
            text_modes=modes,
            compare_to=get_table_data()[question_id][models[0]][files_id[0]],
        )
        table_data.extend(row_data)
    return table_data


def get_general_stats_layout(
    base_model: str,
) -> html.Div:
    data_for_base_model = [data.get(base_model, []) for data in get_table_data()]
    custom_stats = {}
    for name, func in get_general_custom_stats().items():
        errors_dict = {}
        custom_stats[name] = catch_eval_exception(
            [],
            func,
            data_for_base_model,
            "Got error when applying function",
            errors_dict,
        )
        if len(errors_dict):
            logging.error(ERROR_MESSAGE_TEMPLATE.format(name, errors_dict))

    overall_samples = sum(len(question_data) for question_data in data_for_base_model)
    dataset_size = len(list(filter(lambda x: bool(x), data_for_base_model)))
    stats = {
        "dataset size": dataset_size,
        "overall number of samples": overall_samples,
        "generations per sample": (overall_samples / dataset_size if dataset_size else 0),
        **custom_stats,
    }
    return [html.Div([html.Pre(f'{name}: {value}') for name, value in stats.items()])]


def get_update_dataset_layout(base_model: str, update_function: str, models: List[str]) -> List[html.Tr]:
    errors_dict = {}
    global table_data
    if update_function:
        update_eval_function = get_eval_function(update_function.strip())
        available_models = {
            model_name: model_info["file_paths"] for model_name, model_info in get_available_models().items()
        }

        for question_id in range(len(table_data)):
            new_dicts = list(
                map(
                    lambda data: catch_eval_exception(
                        available_models,
                        update_eval_function,
                        data,
                        data,
                        errors_dict,
                    ),
                    table_data[question_id][base_model],
                )
            )
            for i, new_dict in enumerate(new_dicts):
                for key, value in new_dict.items():
                    table_data[question_id][base_model][i][key] = value

                keys = list(table_data[question_id][base_model][i].keys())
                for key in keys:
                    if key not in new_dict:
                        table_data[question_id][base_model][i].pop(key)

    if len(errors_dict):
        logging.error(ERROR_MESSAGE_TEMPLATE.format("update_dataset", errors_dict))

    return (
        get_stats_layout()
        + get_general_stats_layout(base_model)
        + get_table_answers_detailed_data_layout(
            models,
            list(
                filter(
                    is_detailed_answers_rows_key,
                    (
                        table_data[0][base_model][0].keys()
                        if len(table_data) and len(table_data[0][base_model])
                        else []
                    ),
                )
            ),
        )
    )


def get_sorting_answers_layout(base_model: str, sorting_function: str, models: List[str]) -> List[html.Tr]:
    errors_dict = {}
    global table_data
    if sorting_function:
        sortting_eval_function = get_eval_function(sorting_function.strip())
        available_models = {
            model_name: model_info["file_paths"] for model_name, model_info in get_available_models().items()
        }

        for question_id in range(len(table_data)):
            for model in table_data[question_id].keys():
                table_data[question_id][model].sort(
                    key=lambda data: catch_eval_exception(
                        available_models,
                        sortting_eval_function,
                        data,
                        0,
                        errors_dict,
                    )
                )

        table_data.sort(
            key=lambda single_question_data: tuple(
                map(
                    lambda data: catch_eval_exception(
                        available_models,
                        sortting_eval_function,
                        data,
                        0,
                        errors_dict,
                    ),
                    single_question_data[base_model],
                )
            )
        )
    if len(errors_dict):
        logging.error(ERROR_MESSAGE_TEMPLATE.format("sorting", errors_dict))

    return (
        get_stats_layout()
        + get_general_stats_layout(base_model)
        + get_table_answers_detailed_data_layout(
            models,
            list(
                filter(
                    is_detailed_answers_rows_key,
                    (
                        table_data[0][base_model][0].keys()
                        if len(table_data) and len(table_data[0][base_model])
                        else []
                    ),
                )
            ),
        )
    )


def get_filter_answers_layout(
    base_model: str,
    filtering_function: str,
    apply_on_filtered_data: bool,
    models: List[str],
    filter_mode: str,
) -> List[html.Tr]:
    global table_data
    clean_table_data = []
    if not apply_on_filtered_data:
        table_data = custom_deepcopy(get_data_from_files())
        for question_id in range(len(table_data)):
            for model_id, files_data in table_data[question_id].items():
                stats = get_metrics(files_data)
                table_data[question_id][model_id] = list(
                    map(
                        lambda data: {**data, **stats},
                        table_data[question_id][model_id],
                    )
                )

    errors_dict = {}
    if filtering_function:
        available_models = {
            model_name: model_info["file_paths"] for model_name, model_info in get_available_models().items()
        }
        filter_lines = filtering_function.strip().split('\n')
        common_expressions, splitted_filters = (
            "\n".join(filter_lines[:-1]),
            filter_lines[-1],
        )
        full_splitted_filters = [
            common_expressions + "\n" + single_filter for single_filter in splitted_filters.split('&&')
        ]
        filtering_functions = (
            list(
                [
                    get_eval_function(f"{NAME_FOR_BASE_MODEL} = '{base_model}'\n" + func)
                    for func in full_splitted_filters
                ]
            )
            if filtering_function
            else []
        )

        if filter_mode == FILES_FILTERING:
            for question_id in range(len(table_data)):
                good_data = True
                for model_id in table_data[question_id].keys():

                    def filtering_key_function(file_dict):
                        data = {model_id: file_dict}
                        return all(
                            [
                                catch_eval_exception(
                                    available_models,
                                    filter_function,
                                    data,
                                    True,
                                    errors_dict,
                                )
                                for filter_function in filtering_functions
                            ],
                        )

                    table_data[question_id][model_id] = list(
                        filter(
                            filtering_key_function,
                            table_data[question_id][model_id],
                        )
                    )
                    stats = get_metrics(table_data[question_id][model_id])
                    table_data[question_id][model_id] = list(
                        map(
                            lambda data: {**data, **stats},
                            table_data[question_id][model_id],
                        )
                    )

                    if table_data[question_id][model_id] == []:
                        good_data = False
                if good_data:
                    clean_table_data.append(table_data[question_id])
        else:
            func = get_eval_function(f"{NAME_FOR_BASE_MODEL} = '{base_model}'\n" + filtering_function.strip())
            clean_table_data = list(
                filter(
                    lambda data: catch_eval_exception(
                        available_models=[],
                        eval_func=func,
                        data=data,
                        default_answer=True,
                        errors_dict=errors_dict,
                    ),
                    table_data,
                )
            )
        table_data = clean_table_data
    if len(errors_dict):
        logging.error(ERROR_MESSAGE_TEMPLATE.format("filtering", errors_dict))

    return (
        get_stats_layout()
        + get_general_stats_layout(base_model)
        + get_table_answers_detailed_data_layout(
            models,
            list(
                filter(
                    is_detailed_answers_rows_key,
                    (
                        table_data[0][base_model][0].keys()
                        if len(table_data) and len(table_data[0][base_model])
                        else []
                    ),
                )
            ),
        )
    )


def get_model_answers_table_layout(base_model: str, use_current: bool = False) -> List:
    global table_data
    if not use_current:
        table_data = custom_deepcopy(get_data_from_files())

    return (
        get_stats_layout()
        + get_general_stats_layout(base_model)
        + get_table_answers_detailed_data_layout(
            [base_model],
            list(
                filter(
                    is_detailed_answers_rows_key,
                    (
                        table_data[0][base_model][0].keys()
                        if len(table_data) and len(table_data[0][base_model])
                        else []
                    ),
                )
            ),
        )
    )
