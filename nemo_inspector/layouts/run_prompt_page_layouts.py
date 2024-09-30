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

from typing import List, Tuple

import dash_bootstrap_components as dbc
from dash import dcc, html
from flask import current_app
from layouts.base_layouts import get_text_area_layout
from settings.constants import CHAT_MODE, FEW_SHOTS_INPUT, ONE_SAMPLE_MODE
from utils.strategies.strategy_maker import RunPromptStrategyMaker

from nemo_skills.prompt.few_shot_examples import examples_map


def get_few_shots_by_id_layout(page: int, examples_type: str, text_modes: List[str]) -> Tuple[html.Div]:
    examples_list = examples_map.get(examples_type, [{}])
    if not page or len(examples_list) < page:
        return html.Div()
    return (
        html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(key),
                        get_text_area_layout({"type": FEW_SHOTS_INPUT, "id": key}, str(value), text_modes),
                    ],
                    className="mb-3",
                )
                for key, value in (examples_list[page - 1].items())
            ],
        ),
    )


def get_query_params_layout(mode: str = ONE_SAMPLE_MODE, dataset: str = None) -> List[dbc.AccordionItem]:
    strategy = RunPromptStrategyMaker(mode).get_strategy()
    return (
        strategy.get_utils_input_layout()
        + strategy.get_few_shots_input_layout()
        + strategy.get_query_input_layout(dataset)
    )


def get_run_mode_layout() -> html.Div:
    return html.Div(
        [
            dbc.RadioItems(
                id="run_mode_options",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "Chat", "value": CHAT_MODE},
                    {"label": "Run one sample", "value": ONE_SAMPLE_MODE},
                ],
                value=ONE_SAMPLE_MODE,
            ),
        ],
        className="radio-group",
    )


def get_run_test_layout() -> html.Div:
    return html.Div(
        [
            get_run_mode_layout(),
            dbc.Accordion(
                get_query_params_layout(dataset=current_app.config['nemo_inspector']['input_file']),
                start_collapsed=True,
                always_open=True,
                id="prompt_params_input",
            ),
            dbc.Button(
                "preview",
                id="preview_button",
                outline=True,
                color="primary",
                className="me-1 mb-2",
            ),
            dbc.Button(
                "run",
                id="run_button",
                outline=True,
                color="primary",
                className="me-1 mb-2",
            ),
            dcc.Loading(
                children=dbc.Container(id="loading_container", style={'display': 'none'}, children=""),
                type='circle',
                style={'margin-top': '50px'},
            ),
            dbc.Container(id="results_content"),
        ]
    )
