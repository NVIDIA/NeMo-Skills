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
import os
from typing import Dict, List, Tuple

from callbacks import app
from dash import ALL, callback_context, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import current_app
from layouts import (
    get_detailed_answer_column,
    get_filter_answers_layout,
    get_labels,
    get_model_answers_table_layout,
    get_models_selector_table_cell,
    get_row_detailed_inner_data,
    get_sorting_answers_layout,
    get_table_data,
    get_table_detailed_inner_data,
)
from settings.constants import (
    CHOOSE_LABEL,
    CHOOSE_MODEL,
    DELETE,
    ERROR_MESSAGE_TEMPLATE,
    EXTRA_FIELDS,
    GENERAL_STATS,
    LABEL,
    LABEL_SELECTOR_ID,
    MODEL_SELECTOR_ID,
)
from utils.common import (
    calculate_metrics_for_whole_data,
    get_available_models,
    get_custom_stats,
    get_deleted_stats,
    get_excluded_row,
    get_filtered_files,
    get_general_custom_stats,
)

from visualization.layouts.analyze_page_layouts import get_stats_input


@app.callback(
    [
        Output("compare_models_rows", "children", allow_duplicate=True),
        Output("loading_container", "children", allow_duplicate=True),
    ],
    Input("base_model_answers_selector", "value"),
    State("loading_container", "children"),
    prevent_initial_call=True,
)
def choose_base_model(
    base_model: str,
    loading_container: str,
) -> Tuple[List, bool]:
    if base_model == CHOOSE_MODEL:
        return no_update, no_update
    get_excluded_row().clear()
    return (
        get_model_answers_table_layout(
            base_model=base_model,
        ),
        loading_container + " ",
    )


@app.callback(
    [
        Output("save_dataset_modal", "children", allow_duplicate=True),
        Output("save_dataset_modal", "is_open"),
    ],
    [
        Input("save_dataset", "n_clicks"),
    ],
    [State("base_model_answers_selector", "value")],
    prevent_initial_call=True,
)
def save_dataset(n_click: int, base_model: str) -> Tuple[List, bool]:
    if (
        not n_click
        or not current_app.config['data_explorer']['visualization_params']['save_dataset_path']
        or not base_model
    ):
        return no_update
    path = current_app.config['data_explorer']['visualization_params']['save_dataset_path']
    if not os.path.exists(path):
        os.mkdir(path)

    new_data = {}

    for data in get_table_data():
        for file_data in data[base_model]:
            file_name = file_data['file_name']
            if file_name not in new_data:
                new_data[file_name] = []
            for field in EXTRA_FIELDS:
                file_data.pop(field)
            new_data[file_name].append(file_data)

    for file_name, data in new_data.items():
        with open(os.path.join(path, file_name + '.jsonl'), 'w') as file:
            file.write("\n".join([json.dumps(line) for line in data]))

    return f'dataset is saved into {path}', True


@app.callback(
    [
        Output({"type": "filter", "id": ALL}, "is_open"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "set_filter_button", "id": ALL}, "n_clicks"),
        Input({"type": "apply_filter_button", "id": ALL}, "n_clicks"),
    ],
    [
        State({"type": "filter", "id": ALL}, "is_open"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def toggle_modal_filter(n1: int, n2: int, is_open: bool, js_trigger: str) -> bool:
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(is_open), no_update, no_update
    button_id = json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id'] + 1

    if not ctx.triggered[0]['value']:
        return [no_update] * len(is_open), "", js_trigger + " "

    if n1[button_id] or n2[button_id]:
        is_open[button_id] = not is_open[button_id]
        return is_open, "", js_trigger + " "
    return is_open, "", js_trigger + " "


@app.callback(
    [
        Output({"type": "sorting", "id": ALL}, "is_open"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "set_sorting_button", "id": ALL}, "n_clicks"),
        Input({"type": "apply_sorting_button", "id": ALL}, "n_clicks"),
    ],
    [
        State({"type": "sorting", "id": ALL}, "is_open"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def toggle_modal_sorting(n1: int, n2: int, is_open: bool, js_trigger: str) -> bool:
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(is_open), no_update, no_update

    button_id = json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id'] + 1

    if not ctx.triggered[0]['value']:
        return [no_update] * len(is_open), no_update, no_update

    if n1[button_id] or n2[button_id]:
        is_open[button_id] = not is_open[button_id]
        return is_open, "", js_trigger + " "
    return is_open, "", js_trigger + " "


@app.callback(
    Output({"type": "label", "id": ALL}, "is_open"),
    [
        Input({"type": "set_file_label_button", "id": ALL}, "n_clicks"),
        Input({"type": "apply_label_button", "id": ALL}, "n_clicks"),
        Input({"type": "delete_label_button", "id": ALL}, "n_clicks"),
    ],
    [State({"type": "label", "id": ALL}, "is_open")],
)
def toggle_modal_label(n1: int, n2: int, n3: int, is_open: bool) -> bool:
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(is_open)

    button_id = json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id'] + 1
    if not ctx.triggered[0]['value']:
        return [no_update] * len(is_open)

    if n1[button_id] or n2[button_id] or n3[button_id]:
        is_open[button_id] = not is_open[button_id]
        return is_open
    return is_open


@app.callback(
    [
        Output("new_stats", "is_open"),
        Output("stats_input_container", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("set_new_stats_button", "n_clicks"),
        Input("apply_new_stats", "n_clicks"),
    ],
    [
        State("new_stats", "is_open"),
        State("stats_modes", "value"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def toggle_modal_stats(n1: int, n2: int, is_open: bool, modes: List[str], js_trigger: str) -> bool:
    if not n1 and not n2:
        return no_update, no_update, no_update, no_update

    if n1 or n2:
        is_open = not is_open
        return is_open, get_stats_input(modes), "", js_trigger + " "
    return is_open, get_stats_input(modes), "", js_trigger + " "


@app.callback(
    Output("compare_models_rows", "children", allow_duplicate=True),
    Input("apply_new_stats", "n_clicks"),
    [
        State("stats_input", "value"),
        State("base_model_answers_selector", "value"),
        State("stats_modes", "value"),
    ],
    prevent_initial_call=True,
)
def apply_new_stat(
    n_click: int,
    code_raw: str,
    base_model: str,
    stats_modes: List[str],
) -> List:
    if not n_click or code_raw == "":
        return no_update
    code_raw_lines = code_raw.strip().split('\n')
    if not stats_modes or DELETE not in stats_modes:
        code = '\n'.join(code_raw_lines[:-1]) + '\nnew_stats = ' + code_raw_lines[-1]
    else:
        code = "delete_stats = " + f"'{code_raw_lines[-1]}'"
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        logging.error(ERROR_MESSAGE_TEMPLATE.format(code, str(e)))
        return no_update
    if stats_modes and GENERAL_STATS in stats_modes:
        if DELETE in stats_modes:
            get_general_custom_stats().pop(namespace['delete_stats'], None)
        else:
            get_general_custom_stats().update(namespace['new_stats'])
    else:
        if stats_modes and DELETE in stats_modes:
            get_custom_stats().pop(namespace['delete_stats'], None)
            get_deleted_stats().update(namespace['delete_stats'])
        else:
            get_custom_stats().update(namespace['new_stats'])
    if base_model == CHOOSE_MODEL:
        return []
    calculate_metrics_for_whole_data(get_table_data(), base_model)
    return get_model_answers_table_layout(base_model=base_model, use_current=True)


@app.callback(
    [
        Output("compare_models_rows", "children", allow_duplicate=True),
        Output("filtering_container", "children"),
        Output("loading_container", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "apply_filter_button", "id": -1}, "n_clicks"),
    ],
    [
        State({"type": "filter_function_input", "id": -1}, "value"),
        State({"type": "apply_on_filtered_data", "id": -1}, "value"),
        State({"type": "sorting_function_input", "id": -1}, "value"),
        State({"type": "model_selector", "id": ALL}, "value"),
        State("base_model_answers_selector", "value"),
        State("filtering_container", "children"),
        State("loading_container", "children"),
    ],
    prevent_initial_call=True,
)
def filter_data(
    n_ckicks: int,
    filter_function: str,
    apply_on_filtered_data: int,
    sorting_function: str,
    models: List[str],
    base_model: str,
    filtering_functions: str,
    loading_container: str,
) -> Tuple[List[html.Tr], bool]:
    if not n_ckicks:
        return no_update, no_update, no_update
    if apply_on_filtered_data and filtering_functions:
        filtering_functions['props']['children'] += f"\n{filter_function}"
    if base_model == CHOOSE_MODEL:
        return [], no_update, no_update
    if len(get_table_data()) == 0:  # TODO fix
        models = [models[0]]
    get_filter_answers_layout(
        base_model=base_model,
        filtering_function=filter_function,
        apply_on_filtered_data=(apply_on_filtered_data if apply_on_filtered_data else 0),
        models=models,
    )
    return (
        get_sorting_answers_layout(
            base_model=base_model,
            sorting_function=sorting_function,
            models=models,
        ),
        (
            html.Pre(f"Filtering function:\n{filter_function}")
            if not apply_on_filtered_data or not filtering_functions
            else filtering_functions
        ),
        loading_container + " ",
    )


@app.callback(
    [
        Output("compare_models_rows", "children", allow_duplicate=True),
        Output("sorting_container", "children"),
        Output("loading_container", "children", allow_duplicate=True),
    ],
    Input({"type": "apply_sorting_button", "id": -1}, "n_clicks"),
    [
        State({"type": "sorting_function_input", "id": -1}, "value"),
        State({"type": "model_selector", "id": ALL}, "value"),
        State("base_model_answers_selector", "value"),
        State("loading_container", "children"),
    ],
    prevent_initial_call=True,
)
def sorting_data(
    n_ckicks: int,
    sorting_function: str,
    models: List[str],
    base_model: str,
    loading_container: str,
) -> Tuple[List[html.Tr], bool]:
    if base_model == CHOOSE_MODEL or not sorting_function:
        return no_update, no_update, no_update
    return (
        get_sorting_answers_layout(
            base_model=base_model,
            sorting_function=sorting_function,
            models=models,
        ),
        html.Pre(f'Sorting function:\n{sorting_function}'),
        loading_container + " ",
    )


@app.callback(
    [
        Output(
            "dummy_output",
            'children',
            allow_duplicate=True,
        ),
        Output({"type": "del_row", "id": ALL}, "children"),
    ],
    Input({"type": "del_row", "id": ALL}, "n_clicks"),
    [
        State({"type": "row_name", "id": ALL}, "children"),
        State({"type": "del_row", "id": ALL}, "id"),
        State({"type": "del_row", "id": ALL}, "children"),
        State(
            "dummy_output",
            'children',
        ),
    ],
    prevent_initial_call=True,
)
def del_row(
    n_clicks: List[int],
    rows: List[str],
    button_ids: List[Dict],
    del_row_labels: List[str],
    dummy_data: str,
) -> None:
    ctx = callback_context
    if not ctx.triggered or not n_clicks:
        return no_update, [no_update] * len(button_ids)
    button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['id']
    row_index = 0
    for i, current_button_id in enumerate(button_ids):
        if current_button_id['id'] == button_id:
            row_index = i
            break
    if not n_clicks[row_index]:
        return no_update, [no_update] * len(button_ids)
    if rows[row_index] in get_excluded_row():
        get_excluded_row().remove(rows[row_index])
        del_row_labels[row_index] = "-"
    else:
        get_excluded_row().add(rows[row_index])
        del_row_labels[row_index] = "+"

    return dummy_data + '1', del_row_labels


@app.callback(
    Output('datatable', 'data'),
    [
        Input('datatable', "page_current"),
        Input('datatable', "page_size"),
    ],
    [
        State("base_model_answers_selector", "value"),
    ],
)
def change_page(page_current: int, page_size: int, base_model: str) -> List[Dict]:
    if not get_table_data():
        return no_update
    return [
        data[base_model][0]
        for data in get_table_data()[page_current * page_size : (page_current + 1) * page_size]
        if base_model in data.keys()
    ]


@app.callback(
    Output(
        {'type': 'detailed_models_answers', 'id': ALL},
        'children',
        allow_duplicate=True,
    ),
    [
        Input('datatable', 'selected_rows'),
        Input(
            "dummy_output",
            'children',
        ),
    ],
    [
        State({"type": "model_selector", "id": ALL}, "value"),
        State({"type": "sorting_function_input", "id": ALL}, "value"),
        State({"type": "filter_function_input", "id": ALL}, "value"),
        State({"type": "row_name", "id": ALL}, "children"),
        State('datatable', "page_current"),
        State('datatable', "page_size"),
        State({"type": 'file_selector', "id": ALL}, 'value'),
    ],
    prevent_initial_call=True,
)
def show_item(
    idx: List[int],
    dummmy_trigger: str,
    models: List[str],
    sorting_functions: List[str],
    filter_functions: List[str],
    rows_names: List[str],
    current_page: int,
    page_size: int,
    file_names: List[str],
) -> List[str]:
    if not idx:
        raise PreventUpdate
    question_id = current_page * page_size + idx[0]
    file_ids = [0] * len(models)
    for model_id, name in enumerate(file_names):
        for file_id, file in enumerate(get_table_data()[question_id][models[model_id]]):
            if file['file_name'] == name:
                file_ids[model_id] = file_id

    return get_table_detailed_inner_data(
        question_id=question_id,
        rows_names=rows_names,
        models=models,
        files_id=file_ids,
        filter_functions=filter_functions[1:],
        sorting_functions=sorting_functions[1:],
    )


@app.callback(
    [
        Output("stats_input_container", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("stats_modes", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def change_stats_mode(modes: List[str], js_trigger: str) -> str:
    if modes is None:
        return no_update, no_update, no_update
    return get_stats_input(modes), "", js_trigger + " "


@app.callback(
    [
        Output(
            "dummy_output",
            'children',
            allow_duplicate=True,
        ),
    ],
    [
        Input({"type": "apply_label_button", "id": ALL}, "n_clicks"),
        Input({"type": "delete_label_button", "id": ALL}, "n_clicks"),
    ],
    [
        State(
            {"type": "aplly_for_all_files", "id": ALL},
            "value",
        ),
        State({"type": "label_selector", "id": ALL}, 'value'),
        State({"type": "label_selector", "id": ALL}, "id"),
        State('datatable', "page_current"),
        State('datatable', "page_size"),
        State('datatable', 'selected_rows'),
        State({"type": "model_selector", "id": ALL}, "value"),
        State("base_model_answers_selector", "value"),
        State({"type": 'file_selector', "id": ALL}, 'value'),
        State({"type": 'file_selector', "id": ALL}, 'options'),
        State(
            "dummy_output",
            'children',
        ),
    ],
    prevent_initial_call=True,
)
def change_label(
    n_click_apply: List[int],
    n_click_del: List[int],
    apply_for_all: List[bool],
    labels: List[str],
    label_ids: List[int],
    current_page: int,
    page_size: int,
    idx: List[int],
    models: List[str],
    base_model: str,
    file_names: List[str],
    file_options: List[str],
    dummy_data: str,
) -> List[List[str]]:
    ctx = callback_context
    if not ctx.triggered:
        return [no_update]

    button_id = label_ids.index(
        json.loads(LABEL_SELECTOR_ID.format(json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id']))
    )
    is_apply = json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['type'] == "apply_label_button"
    if not ctx.triggered[0]['value'] or labels[button_id] == CHOOSE_LABEL:
        return [no_update]

    ALL_FILES = "ALL_FILES"
    if button_id == 0:
        files = [ALL_FILES]
        file = [ALL_FILES]
        models_to_process = [(base_model, files, file)]
        apply_for_all = [[True] * len(models)]
        question_ids = list(range(len(get_table_data())))
    else:
        if not idx:
            return [no_update]
        models_to_process = [
            (
                models[button_id - 1],
                file_options[button_id - 1],
                file_names[button_id - 1],
            )
        ]
        question_ids = [current_page * page_size + idx[0]]

    apply_for_all_files = bool(len(apply_for_all[button_id - 1]))
    for question_id in question_ids:
        for model, current_file_options, current_file in models_to_process:
            options = (
                current_file_options
                if button_id != 0
                else [{'value': file['file_name']} for file in get_table_data()[question_id][model]]
            )
            for file in options:
                if not apply_for_all_files and not file['value'] == current_file:
                    continue

                file_id = 0
                for i, model_file in enumerate(get_table_data()[question_id][model]):
                    if model_file['file_name'] == file['value']:
                        file_id = i
                        break

                if labels[button_id] not in get_table_data()[question_id][model][file_id][LABEL]:
                    if is_apply:
                        get_table_data()[question_id][model][file_id][LABEL].append(labels[button_id])

                elif not is_apply:
                    get_table_data()[question_id][model][file_id][LABEL].remove(labels[button_id])

    return [dummy_data + "1"]


@app.callback(
    Output(
        {'type': 'detailed_models_answers', 'id': ALL},
        'children',
        allow_duplicate=True,
    ),
    [
        Input({"type": 'file_selector', "id": ALL}, 'value'),
        Input({"type": 'plain_text_switch', "id": ALL}, 'value'),
    ],
    [
        State('datatable', 'selected_rows'),
        State({"type": "sorting_function_input", "id": ALL}, "value"),
        State({"type": "filter_function_input", "id": ALL}, "value"),
        State({"type": "model_selector", "id": ALL}, "value"),
        State({"type": "model_selector", "id": ALL}, "id"),
        State({"type": "row_name", "id": ALL}, "children"),
        State('datatable', "page_current"),
        State('datatable', "page_size"),
        State(
            {'type': 'detailed_models_answers', 'id': ALL},
            'children',
        ),
    ],
    prevent_initial_call=True,
)
def change_file(
    file_names: List[str],
    plain_text_switch: List[str],
    idx: List[int],
    sorting_functions: List[str],
    filter_functions: List[str],
    models: List[str],
    model_ids: List[int],
    rows_names: List[str],
    current_page: int,
    page_size: int,
    table_data: List[str],
) -> List[str]:
    if not idx:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(table_data)

    question_id = page_size * current_page + idx[0]
    try:
        button_id = model_ids.index(
            json.loads(MODEL_SELECTOR_ID.format(json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id']))
        )
    except ValueError:
        return [no_update] * len(table_data)

    model = models[button_id]

    filtered_files = get_filtered_files(
        filter_function=filter_functions[button_id + 1],
        sorting_function=sorting_functions[button_id + 1],
        array_to_filter=get_table_data()[question_id][model],
    )
    file_id = 0
    for i, file_data in enumerate(filtered_files):
        if file_data['file_name'] == file_names[button_id]:
            file_id = i
            break

    question_id = current_page * page_size + idx[0]
    table_data[button_id * len(rows_names) : (button_id + 1) * len(rows_names)] = get_row_detailed_inner_data(
        question_id=question_id,
        model=model,
        file_id=file_id,
        rows_names=rows_names,
        col_id=button_id,
        filter_function=filter_functions[button_id + 1],
        sorting_function=sorting_functions[button_id + 1],
        plain_text=(plain_text_switch[button_id] and len(plain_text_switch[button_id])),
    )
    return table_data


@app.callback(
    [
        Output({"type": "new_label_input", "id": ALL}, "value"),
        Output({"type": "label_selector", "id": ALL}, "options"),
        Output({"type": "label_selector", "id": ALL}, 'value'),
    ],
    Input({"type": "add_new_label_button", "id": ALL}, "n_clicks"),
    [
        State({"type": "new_label_input", "id": ALL}, "value"),
        State({"type": "label_selector", "id": ALL}, "options"),
        State({"type": "label_selector", "id": ALL}, 'value'),
        State({"type": "label_selector", "id": ALL}, "id"),
    ],
)
def add_new_label(
    n_click: int,
    new_labels: List[str],
    options: List[List[str]],
    values: List[str],
    label_ids: List[int],
) -> Tuple[List[List[str]], List[str]]:
    ctx = callback_context
    no_updates = [no_update] * len(new_labels)
    if not ctx.triggered:
        return no_updates, no_updates, no_updates

    button_id = label_ids.index(
        json.loads(LABEL_SELECTOR_ID.format(json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id']))
    )

    if not ctx.triggered[0]['value']:
        return no_updates, no_updates, no_updates

    if new_labels[button_id] and new_labels[button_id] not in options[button_id]:
        for i in range(len(options)):
            new_label = {'label': new_labels[button_id], 'value': new_labels[button_id]}
            if new_label not in options[i]:
                options[i].append({'label': new_labels[button_id], 'value': new_labels[button_id]})
        values[button_id] = new_labels[button_id]
    else:
        return no_updates, no_updates, no_updates

    get_labels().append(new_labels[button_id])
    new_labels[button_id] = ""

    return new_labels, options, values


@app.callback(
    Output({"type": "chosen_label", "id": ALL}, "children"),
    Input({"type": "label_selector", "id": ALL}, "value"),
    [
        State({"type": "label_selector", "id": ALL}, "id"),
        State({"type": "chosen_label", "id": ALL}, "children"),
    ],
)
def choose_label(
    label: List[str], label_ids: List[int], chosen_labels: List[str]
) -> Tuple[List[List[str]], List[str]]:
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(chosen_labels)

    for trigger in ctx.triggered:
        button_id = label_ids.index(
            json.loads(LABEL_SELECTOR_ID.format(json.loads(trigger['prop_id'].split('.')[0])['id']))
        )

        if not ctx.triggered[0]['value'] or label[button_id] == CHOOSE_LABEL:
            chosen_labels[button_id] = ""
        else:
            chosen_labels[button_id] = f"chosen label: {label[button_id]}"

    return chosen_labels


@app.callback(
    [
        Output(
            "detailed_answers_header",
            "children",
            allow_duplicate=True,
        ),
        Output(
            {"type": "detailed_answers_row", "id": ALL},
            "children",
            allow_duplicate=True,
        ),
    ],
    Input("add_model", "n_clicks"),
    [
        State("detailed_answers_header", "children"),
        State({"type": "detailed_answers_row", "id": ALL}, "children"),
        State({"type": "model_selector", "id": ALL}, "id"),
        State("datatable", "selected_rows"),
    ],
    prevent_initial_call=True,
)
def add_model(
    n_clicks: int,
    header: List,
    rows: List,
    selectors_ids: List[int],
    idx: List[int],
) -> Tuple[List, List]:
    if not n_clicks:
        return no_update, [no_update] * len(rows)
    available_models = list(get_available_models().keys())
    last_header_id = selectors_ids[-1]['id'] if selectors_ids != [] else -1
    header.append(get_models_selector_table_cell(available_models, available_models[0], last_header_id + 1, True))
    last_cell_id = rows[-1][-1]["props"]["children"]["props"]['id']['id']
    for i, row in enumerate(rows):
        row.append(
            get_detailed_answer_column(
                last_cell_id + i + 1,
                file_id=last_header_id + 1 if i == 0 and idx else None,
            )
        )

    return header, rows


@app.callback(
    [
        Output("detailed_answers_header", "children"),
        Output({"type": "detailed_answers_row", "id": ALL}, "children"),
    ],
    Input({"type": "del_model", "id": ALL}, "n_clicks"),
    [
        State("detailed_answers_header", "children"),
        State({"type": "detailed_answers_row", "id": ALL}, "children"),
        State({"type": "del_model", "id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def del_model(
    n_clicks: List[int],
    header: List,
    rows: List,
    id_del: List[int],
) -> Tuple[List, List]:
    ctx = callback_context
    if not ctx.triggered:
        return no_update, [no_update] * len(rows)

    button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['id']

    if not ctx.triggered[0]['value']:
        return no_update, [no_update] * len(rows)

    for i, id in enumerate(id_del):
        if id['id'] == button_id:
            index = i + 2

    header.pop(index)
    for i, row in enumerate(rows):
        row.pop(index)

    return header, rows


@app.callback(
    [
        Output({"type": 'file_selector', "id": ALL}, 'options'),
        Output({"type": 'file_selector', "id": ALL}, 'value'),
    ],
    [
        Input({"type": "apply_filter_button", "id": ALL}, "n_clicks"),
        Input({"type": "apply_sorting_button", "id": ALL}, "n_clicks"),
        Input({"type": "model_selector", "id": ALL}, "value"),
    ],
    [
        State({"type": "model_selector", "id": ALL}, "id"),
        State({"type": "sorting_function_input", "id": ALL}, "value"),
        State({"type": "filter_function_input", "id": ALL}, "value"),
        State({"type": "apply_on_filtered_data", "id": ALL}, "value"),
        State('datatable', "page_current"),
        State('datatable', "page_size"),
        State('datatable', 'selected_rows'),
        State({"type": 'file_selector', "id": ALL}, 'options'),
        State({"type": 'file_selector', "id": ALL}, 'value'),
    ],
    prevent_initial_call=True,
)
def change_files_order(
    filter_n_click: int,
    sorting_n_click: int,
    models: List[str],
    model_ids: List[int],
    sorting_functions: List[str],
    filter_functions: List[str],
    apply_on_filtered_data: List[int],
    current_page: int,
    page_size: int,
    idx: List[int],
    file_selector_options: List[str],
    file_selector_values: List[str],
) -> Tuple[List[List[str]], List[str]]:
    no_updates = [no_update] * len(file_selector_options)
    if not filter_n_click and not sorting_n_click:
        return no_updates, no_updates
    if not idx:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        return no_updates, no_updates

    try:
        button_id = model_ids.index(
            json.loads(MODEL_SELECTOR_ID.format(json.loads(ctx.triggered[-1]['prop_id'].split('.')[0])['id']))
        )
    except ValueError:
        return no_updates, no_updates

    if not ctx.triggered[0]['value'] or button_id == -1:
        return no_updates, no_updates
    model = models[button_id]
    question_id = current_page * page_size + idx[0]
    array_to_filter = (
        get_table_data()[question_id][model]
        if not apply_on_filtered_data or not apply_on_filtered_data[button_id]
        else list(
            filter(
                lambda data: data['file_name'] in [file_name['label'] for file_name in file_selector_options],
                get_table_data()[question_id][model],
            )
        )
    )
    file_selector_options[button_id] = [
        {'label': data['file_name'], 'value': data['file_name']}
        for data in get_filtered_files(
            filter_functions[button_id + 1],
            sorting_functions[button_id + 1],
            array_to_filter,
        )
    ]
    file_selector_values[button_id] = file_selector_options[button_id][0]

    return file_selector_options, file_selector_values
