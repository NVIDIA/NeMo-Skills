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
from typing import List

import dash_bootstrap_components as dbc
from dash import dash_table, html
from layouts.base_layouts import get_selector_layout, get_single_prompt_output_layout, get_switch_layout
from settings.constants import DATA_PAGE_SIZE, ERROR_MESSAGE_TEMPLATE, MODEL_SELECTOR_ID, STATS_KEYS
from utils.common import (
    catch_eval_exception,
    custom_deepcopy,
    get_available_models,
    get_data_from_files,
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


def get_filter_layout(id: int = -1, available_filters: List[str] = [], files_only: bool = False) -> html.Div:
    available_filters = list(
        get_table_data()[0][list(get_table_data()[0].keys())[0]][0].keys()
        if len(get_table_data()) and not available_filters
        else STATS_KEYS + list(get_metrics([]).keys()) + ["+ all fields in json"]
    )

    inner_text = (
        "Separate expressions for different models with &&\n\n"
        + "For example:\ndata['model1']['correct_responses'] > 0.5 && data['model2']['no_response'] < 0.2\n\n"
        if not files_only
        else "For example:\ndata['correct_responses'] > 0.5 and data['no_response'] < 0.2\n\n"
    )
    text = (
        "Write an expression to filter the data\n" + inner_text + "The function has to return bool.\n\n"
        "Available parameters to filter data:\n"
        + '\n'.join([', '.join(available_filters[start : start + 5]) for start in range(0, len(available_filters), 5)])
    )
    header = dbc.ModalHeader(dbc.ModalTitle("Set Up Your Filter"), close_button=True)
    body = dbc.ModalBody(
        html.Div(
            [
                html.Pre(text),
                dbc.Textarea(
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
                style={'margin-left': '2px'},
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
                dbc.Textarea(
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
                style={'margin-left': '2px'},
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
                style={'margin-left': '2px'},
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
                get_filter_layout(id, files_only=True),
                get_change_label_layout(id),
            ]
            + del_model_layout
            + [
                get_switch_layout(
                    {
                        "type": "plain_text_switch",
                        "id": id,
                    },
                    ["plain text"],
                )
            ],
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
                                    "-",
                                    id={"type": "del_row", "id": i},
                                    outline=True,
                                    color="primary",
                                    className="me-1",
                                    style={
                                        "border": "none",
                                        "display": "inline-block",
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
    file_id: int,
    col_id: int,
    filter_function: str = "",
    sorting_function: str = "",
    plain_text: bool = False,
) -> List:
    table_data = get_filtered_files(
        filter_function=filter_function,
        sorting_function=sorting_function,
        array_to_filter=get_table_data()[question_id].get(model, []),
    )
    files_name = [data['file_name'] for data in table_data]
    row_data = []
    for key in filter(
        lambda key: is_detailed_answers_rows_key(key),
        map(lambda data: data, rows_names),
    ):
        if len(table_data) <= file_id or key in get_excluded_row():
            value = ""
        elif key == 'file_name':
            value = get_selector_layout(
                files_name,
                {"type": "file_selector", "id": col_id},
                table_data[file_id].get(key, None),
            )
        else:
            value = (
                get_single_prompt_output_layout(str(table_data[file_id].get(key, None)))
                if not plain_text
                else html.Pre(str(table_data[file_id].get(key, None)))
            )
        row_data.append(value)
    return row_data


def get_table_detailed_inner_data(
    question_id: int,
    rows_names: List[str],
    models: List[str],
    files_id: List[int],
    filter_functions: List[str],
    sorting_functions: List[str],
) -> List:
    table_data = []
    for col_id, (
        model,
        file_id,
        filter_function,
        sorting_function,
    ) in enumerate(zip(models, files_id, filter_functions, sorting_functions)):
        row_data = get_row_detailed_inner_data(
            question_id=question_id,
            model=model,
            rows_names=rows_names,
            file_id=file_id,
            col_id=col_id,
            filter_function=filter_function,
            sorting_function=sorting_function,
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
    stats = {
        "overall number of samples": sum(len(question_data) for question_data in data_for_base_model),
        **custom_stats,
    }
    return [html.Div([html.Div(f'{name}: {value}') for name, value in stats.items()])]


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

        filtering_functions = (
            list([get_eval_function(func.strip()) for func in filtering_function.split('&&')])
            if filtering_function
            else []
        )
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
