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
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
from callbacks import app
from dash import ALL, callback_context, html, no_update
from dash._callback import NoUpdate
from dash.dependencies import Input, Output, State
from flask import current_app
from layouts import (
    get_few_shots_by_id_layout,
    get_query_params_layout,
    get_results_content_layout,
    get_single_prompt_output_layout,
    get_utils_field_representation,
)
from settings.constants import (
    FEW_SHOTS_INPUT,
    QUERY_INPUT_TYPE,
    RETRIEVAL,
    RETRIEVAL_FIELDS,
    SEPARATOR_DISPLAY,
    SEPARATOR_ID,
    UNDEFINED,
)
from utils.common import (
    default_examples,
    extract_query_params,
    get_config,
    get_examples,
    get_settings,
    get_test_data,
    get_utils_dict,
    get_utils_from_config,
    get_values_from_input_group,
)
from utils.strategies.strategy_maker import RunPromptStrategyMaker

from nemo_skills.inference.prompt.utils import (
    FewShotExamplesConfig,
    Prompt,
    PromptConfig,
    context_templates,
    get_prompt_config,
    prompt_types,
)


@app.callback(
    [
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("prompt_params_input", "active_item"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def trigger_js(active_item: str, js_trigger: str) -> Tuple[str, str]:
    return "", js_trigger + " "


@app.callback(
    [
        Output("utils_group", "children", allow_duplicate=True),
        Output("few_shots_div", "children"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("examples_type", "value"),
        Input("num_few_shots", "value"),
        Input('data_file', 'value'),
        Input('retrieve_button', 'n_clicks'),
        Input({"type": RETRIEVAL, "id": ALL}, "value"),
    ],
    [
        State("js_trigger", "children"),
        State('utils_group', 'children'),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def update_examples_type(
    examples_type: str,
    num_few_shots: int,
    data_file: str,
    retrieve_n_click: int,
    retrieval_fields: List,
    js_trigger: str,
    raw_utils: List[Dict],
    query_params: List[str],
    query_params_ids: List[Dict],
) -> Union[NoUpdate, dbc.AccordionItem]:
    if not examples_type:
        examples_type = ""
    if retrieve_n_click:
        for key, value in default_examples.items():
            if key == examples_type:
                get_examples()[examples_type] = deepcopy(value)
                break
    data_file_index = 0
    retrieval_field_index = -1

    for retrieval_index, util in enumerate(raw_utils):
        name = util['props']['children'][0]['props']['children']
        if name == 'data_file':
            data_file_index = retrieval_index
        if name == 'retrieval_field':
            retrieval_field_index = retrieval_index

    if examples_type == RETRIEVAL:
        utils = {key.split(SEPARATOR_ID)[-1]: value for key, value in get_values_from_input_group(raw_utils).items()}
        utils.pop('examples_type', None)
        prompt_config = get_config(PromptConfig, utils, get_settings())

        if (
            'retrieval_file' in utils
            and utils['retrieval_file']
            and os.path.isfile(utils['retrieval_file'])
            and os.path.isfile(data_file)
        ):
            with open(utils['retrieval_file'], 'r') as retrieval_file, open(data_file, 'r') as data_file:
                types = current_app.config['data_explorer']['types']
                sample = {
                    key: value
                    for key, value in json.loads(retrieval_file.readline()).items()
                    if key in json.loads(data_file.readline())
                }
            types['retrieval_field'] = list(filter(lambda key: isinstance(sample[key], str), sample.keys()))
            if retrieval_field_index != -1:
                retrieval_field = raw_utils[retrieval_field_index]['props']['children'][1]['props']
                retrieval_field_value = raw_utils[retrieval_field_index]['props']['children'][1]['props']['value']
                retrieval_field['options'] = types['retrieval_field']
                if retrieval_field_value in types['retrieval_field']:
                    retrieval_field['value'] = retrieval_field_value
                else:
                    retrieval_field['value'] = types['retrieval_field'][0]
                utils["retrieval_field"] = retrieval_field['value']

        if raw_utils[data_file_index + 1]['props']['children'][0]['props']['children'] not in RETRIEVAL_FIELDS:
            for retrieval_field in RETRIEVAL_FIELDS:
                raw_utils.insert(
                    data_file_index + 1,
                    get_utils_dict(
                        retrieval_field,
                        current_app.config['data_explorer']['retrieval_fields'][retrieval_field],
                        {"type": RETRIEVAL, "id": retrieval_field},
                    ),
                )
        try:
            prompt_config.few_shot_examples = get_config(
                FewShotExamplesConfig,
                utils,
                get_settings(),
            )

            prompt = Prompt(config=prompt_config)
            get_examples()[examples_type] = prompt.build_examples_dict(
                extract_query_params(query_params_ids, query_params)
            )
        except (ValueError, KeyError, FileNotFoundError) as e:
            get_examples()[examples_type] = []

    else:
        while (
            data_file_index + 1 < len(raw_utils)
            and raw_utils[data_file_index + 1]['props']['children'][0]['props']['children'] in RETRIEVAL_FIELDS
        ):
            raw_utils.pop(data_file_index + 1)

    size = len(
        get_examples().get(
            examples_type,
            [],
        )
    )
    return (
        raw_utils,
        RunPromptStrategyMaker().get_strategy().get_few_shots_div_layout(min(num_few_shots, size)),
        "",
        js_trigger + " ",
    )


@app.callback(
    [
        Output("few_shots_pagination_content", "children"),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("few_shots_pagination", "active_page"),
        Input(
            {
                "type": "text_modes",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
        Input("dummy_output", "children"),
    ],
    State('examples_type', "value"),
    State("num_few_shots", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def change_examples_page(
    page: int,
    text_modes: List[str],
    dummy_output: str,
    examples_type: str,
    num_few_shots: int,
    js_trigger: str,
) -> Tuple[Tuple[html.Div], int]:
    if not examples_type:
        examples_type = ""
    return (
        get_few_shots_by_id_layout(page, examples_type, num_few_shots, text_modes),
        '',
        js_trigger + '',
    )


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
    ],
    Input("add_example_button", "n_clicks"),
    [
        State('examples_type', "value"),
        State("few_shots_pagination", "max_value"),
    ],
    prevent_initial_call=True,
)
def add_example(n_clicks: int, examples_type: str, last_page: int) -> Tuple[int, int, int]:
    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    examples_type_keys = list(get_examples().keys())[0] if not len(get_examples()[examples_type]) else examples_type
    get_examples()[examples_type].append({key: "" for key in get_examples()[examples_type_keys][0].keys()})
    return (last_page + 1, last_page + 1)


@app.callback(
    [
        Output("few_shots_pagination", "max_value", allow_duplicate=True),
        Output("few_shots_pagination", "active_page", allow_duplicate=True),
        Output("few_shots_pagination_content", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("del_example_button", "n_clicks"),
    [
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State("few_shots_pagination", "max_value"),
        State(
            {
                "type": "text_modes",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def del_example(
    n_clicks: int,
    page: int,
    examples_type: str,
    last_page: int,
    text_modes: List[str],
    js_trigger: str,
) -> Tuple[
    Union[int, NoUpdate],
    Union[int, NoUpdate],
    Union[Tuple[html.Div], NoUpdate],
    Union[int, NoUpdate],
]:
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    if last_page:
        prev_pagination_page = page if page < last_page else page - 1
        get_examples()[examples_type].pop(page - 1)
        return (
            last_page - 1,
            prev_pagination_page,
            get_few_shots_by_id_layout(prev_pagination_page, examples_type, last_page - 1, text_modes),
            '',
            js_trigger + ' ',
        )
    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output(
        "dummy_output",
        'children',
        allow_duplicate=True,
    ),
    Input({"type": FEW_SHOTS_INPUT, "id": ALL}, "value"),
    [
        State({"type": FEW_SHOTS_INPUT, "id": ALL}, "id"),
        State("few_shots_pagination", "active_page"),
        State('examples_type', "value"),
        State(
            {
                "type": "text_modes",
                "id": FEW_SHOTS_INPUT,
            },
            "value",
        ),
    ],
    prevent_initial_call=True,
)
def update_examples(
    page_content: Optional[List[str]],
    page_content_ids: List[int],
    page: int,
    examples_type: str,
    text_modes: List[str],
) -> NoUpdate:
    if text_modes and len(text_modes) or not page_content:
        return no_update

    if not examples_type:
        examples_type = ""
    if examples_type not in get_examples():
        get_examples()[examples_type] = []
    last_page = len(get_examples()[examples_type])
    if last_page:
        get_examples()[examples_type][page - 1 if page else 0] = {
            key["id"]: value for key, value in zip(page_content_ids, page_content)
        }
    return no_update


@app.callback(
    [
        Output(
            SEPARATOR_ID.join(field.split(SEPARATOR_DISPLAY)),
            "value",
            allow_duplicate=True,
        )
        for field in get_utils_from_config(
            {"prompt": asdict(PromptConfig(few_shot_examples=FewShotExamplesConfig()))}
        ).keys()
    ]
    + [
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("prompt_type", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def update_prompt_type(prompt_type: str, js_trigger: str) -> Union[NoUpdate, dbc.AccordionItem]:
    if (
        "used_prompt" in current_app.config['data_explorer']['prompt']
        and prompt_type == current_app.config['data_explorer']['prompt']['used_prompt']
    ):
        output_len = len(get_utils_from_config(asdict(PromptConfig(few_shot_examples=FewShotExamplesConfig()))).keys())
        return [no_update] * (output_len + 2)

    current_app.config['data_explorer']['prompt']['used_prompt'] = prompt_type

    if prompt_type not in map(lambda name: name.split('.')[0], prompt_types):
        output_len = len(get_utils_from_config(asdict(PromptConfig(few_shot_examples=FewShotExamplesConfig()))).keys())
        return [no_update] * (output_len + 2)
    prompt_config = get_prompt_config(prompt_type)
    current_app.config['data_explorer']['prompt']['stop_phrases'] = list(prompt_config.stop_phrases)
    return [
        get_utils_field_representation(value, key)
        for key, value in get_utils_from_config(asdict(prompt_config)).items()
    ] + ['', js_trigger + " "]


@app.callback(
    [
        Output("context_template", "value", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    Input("context_type", "value"),
    State("js_trigger", "children"),
    prevent_initial_call=True,
)
def update_context_type(context_type: str, js_trigger: str) -> Union[NoUpdate, dbc.AccordionItem]:
    return context_templates.get(context_type, no_update), "", js_trigger + " "


@app.callback(
    Output("utils_group", "children"),
    Input("range_random_seed_mode", "value"),
    State("utils_group", "children"),
)
def update_random_seed_mode(
    range_random_mode: List[int],
    utils: List[Dict],
) -> List[Dict]:
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    for i, util in enumerate(utils):
        name = util['props']['children'][0]['props']['children']
        if name in ("random_seed", "start_random_seed", "end_random_seed"):
            break
    value = utils[i]['props']['children'][1]['props']['value']
    if range_random_mode:
        utils[i] = get_utils_dict("start_random_seed", value)
        utils.insert(i + 1, get_utils_dict("end_random_seed", value + 1))
    else:
        utils[i] = get_utils_dict("random_seed", value)
        utils.pop(i + 1)
    return utils


@app.callback(
    [
        Output("results_content", "children"),
        Output("loading_container", "children", allow_duplicate=True),
    ],
    Input("run_button", "n_clicks"),
    [
        State("utils_group", "children"),
        State("range_random_seed_mode", "value"),
        State("run_mode_options", "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
        State({"type": "query_store", "id": ALL}, "data"),
        State("loading_container", "children"),
    ],
    prevent_initial_call=True,
)
def get_run_test_results(
    n_clicks: int,
    utils: List[Dict],
    range_random_mode: List[int],
    run_mode: str,
    query_params: List[str],
    query_params_ids: List[Dict],
    query_store: List[Dict[str, str]],
    loading_container: str,
) -> Union[Tuple[html.Div, str], Tuple[NoUpdate, NoUpdate]]:
    if n_clicks is None:
        return no_update, no_update

    utils = get_values_from_input_group(utils)
    if "examples_type" in utils and utils["examples_type"] is None:
        utils["examples_type"] = ""

    if None not in query_params:
        query_store = [extract_query_params(query_params_ids, query_params)]

    return (
        RunPromptStrategyMaker(run_mode)
        .get_strategy()
        .run(
            utils,
            {
                **query_store[0],
                "range_random_mode": range_random_mode,
            },
        ),
        loading_container + " ",
    )


@app.callback(
    [
        Output("prompt_params_input", "children", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
        Output("results_content", "children", allow_duplicate=True),
    ],
    Input("run_mode_options", "value"),
    [
        State("utils_group", "children"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def change_mode(run_mode: str, utils: List[Dict], js_trigger: str) -> Tuple[List[dbc.AccordionItem], None]:
    utils = get_values_from_input_group(utils)
    return (
        get_query_params_layout(run_mode, utils.get('data_file', UNDEFINED)),
        "",
        js_trigger + ' ',
        None,
    )


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
        Output({"type": "query_store", "id": ALL}, "data", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input("query_search_button", "n_clicks"),
        Input("data_file", "value"),
        Input("run_mode_options", "value"),
    ],
    [
        State("query_search_input", "value"),
        State(
            {
                "type": "text_modes",
                "id": QUERY_INPUT_TYPE,
            },
            "value",
        ),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def prompt_search(
    n_clicks: int,
    data_file: str,
    run_mode: str,
    index: int,
    text_modes: List[str],
    js_trigger: str,
) -> Tuple[Union[List[str], NoUpdate]]:
    query_data = get_test_data(index, data_file)[0]
    return (
        RunPromptStrategyMaker()
        .get_strategy()
        .get_query_input_children_layout(
            query_data,
            text_modes,
        ),
        [query_data],
        "",
        js_trigger + " ",
    )


@app.callback(
    [
        Output("query_input_children", "children", allow_duplicate=True),
        Output({"type": "query_store", "id": ALL}, "data", allow_duplicate=True),
        Output("js_container", "children", allow_duplicate=True),
        Output("js_trigger", "children", allow_duplicate=True),
    ],
    [
        Input(
            {
                "type": "text_modes",
                "id": QUERY_INPUT_TYPE,
            },
            "value",
        ),
    ],
    [
        State({"type": "query_store", "id": ALL}, "data"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
        State("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def change_prompt_search_mode(
    text_modes: List[str],
    query_store: List[Dict[str, str]],
    query_params: List[str],
    query_params_ids: List[int],
    js_trigger: str,
) -> Tuple[Union[List[str], NoUpdate]]:
    if None not in query_params:
        query_store = [extract_query_params(query_params_ids, query_params)]

    return (
        RunPromptStrategyMaker()
        .get_strategy()
        .get_query_input_children_layout(
            query_store[0],
            text_modes,
        ),
        query_store,
        "",
        js_trigger + " ",
    )


@app.callback(
    Output("results_content", "children", allow_duplicate=True),
    Input("preview_button", "n_clicks"),
    [
        State("run_mode_options", "value"),
        State("utils_group", "children"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "value"),
        State({"type": QUERY_INPUT_TYPE, "id": ALL}, "id"),
        State({"type": "query_store", "id": ALL}, "data"),
    ],
    prevent_initial_call=True,
)
def preview(
    n_clicks: int,
    run_mode: str,
    utils: List[Dict],
    query_params: List[str],
    query_params_ids: List[int],
    query_store: List[Dict[str, str]],
) -> html.Pre:
    if None not in query_params:
        query_store = [extract_query_params(query_params_ids, query_params)]

    utils = get_values_from_input_group(utils)

    prompt = RunPromptStrategyMaker(run_mode).get_strategy().get_prompt(utils, query_store[0])
    return get_results_content_layout(str(prompt))


@app.callback(
    Output("results_content_text", "children", allow_duplicate=True),
    Input(
        {
            "type": "text_modes",
            "id": "results_content",
        },
        "value",
    ),
    State("text_store", "data"),
    prevent_initial_call=True,
)
def change_results_content_mode(text_modes: List[str], text: str) -> html.Pre:
    return get_single_prompt_output_layout(text, text_modes) if text_modes and len(text_modes) else text
