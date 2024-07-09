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

import logging
from copy import deepcopy
from dataclasses import asdict, fields
from typing import Callable, Dict, List, Union

import dash_bootstrap_components as dbc
import requests
from dash import dcc, html
from flask import current_app
from layouts import (
    get_input_group_layout,
    get_results_content_layout,
    get_single_prompt_output_layout,
    get_switch_layout,
    get_text_area_layout,
)
from settings.constants import (
    FEW_SHOTS_INPUT,
    QUERY_INPUT_TYPE,
    RETRIEVAL,
    RETRIEVAL_FIELDS,
    SEPARATOR_DISPLAY,
    SEPARATOR_ID,
    SETTING_PARAMS,
)
from utils.common import get_config, get_examples, get_settings, get_utils_from_config

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate_solutions import InferenceConfig
from nemo_skills.inference.prompt.utils import FewShotExamplesConfig, Prompt, PromptConfig
from nemo_skills.inference.server.code_execution_model import get_code_execution_model


class ModeStrategies:
    def __init__(self):
        self.sandbox = None

    def sandbox_init(self):
        if self.sandbox is None:
            self.sandbox = get_sandbox(
                **current_app.config['data_explorer']['sandbox'],
            )

    def get_utils_input_layout(
        self,
        condition: Callable[[str, Union[str, int, float, bool]], bool] = lambda key, value: True,
        disabled: bool = False,
    ) -> List[dbc.AccordionItem]:
        utils = get_utils_from_config(
            {key: value for key, value in current_app.config['data_explorer'].items() if key not in SETTING_PARAMS}
        ).items()
        input_group_layout = html.Div(
            (
                [
                    get_input_group_layout(
                        name,
                        value,
                    )
                    for name, value in sorted(
                        utils,
                        key=lambda item: (
                            1
                            if item[0].split(SEPARATOR_DISPLAY)[-1] in current_app.config['data_explorer']['types']
                            else 0 if not isinstance(item[1], str) else 2
                        ),
                    )
                    if condition(name, value)
                ]
            ),
            id="utils_group",
        )
        utils_group_layout = [
            dbc.AccordionItem(
                html.Div(
                    [
                        get_switch_layout(
                            id="range_random_seed_mode",
                            labels=["use random seed range"],
                            disabled=[disabled],
                        ),
                        input_group_layout,
                    ]
                ),
                title="Utils",
            )
        ]
        return utils_group_layout

    def get_few_shots_input_layout(self) -> List[dbc.AccordionItem]:
        config = current_app.config['data_explorer']
        size = config["prompt"]["few_shot_examples"]["num_few_shots"]
        return [
            dbc.AccordionItem(
                self.get_few_shots_div_layout(size),
                title="Few shots",
                id="few_shots_group",
            )
        ]

    def get_query_input_layout(
        self, query_data: Dict[str, str], is_prompt_search: bool = True
    ) -> List[dbc.AccordionItem]:
        switch_layout = [
            get_switch_layout(
                {
                    "type": "view_mode",
                    "id": QUERY_INPUT_TYPE,
                },
                ["view mode"],
            )
        ]
        search_layout = [self._get_search_prompt_layout()] if is_prompt_search else []
        query_input = [
            html.Div(
                self.get_query_input_children_layout(query_data),
                id="query_input_children",
            )
        ]
        query_store = [dcc.Store(id="query_store", data=query_data)]
        return [
            dbc.AccordionItem(
                html.Div(
                    switch_layout + search_layout + query_input + query_store,
                ),
                title="Input",
                id="query_input_content",
            )
        ]

    def get_query_input_children_layout(
        self, query_data: Dict[str, str], view_mode: bool = False
    ) -> List[dbc.InputGroup]:
        return [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(key),
                    get_text_area_layout(
                        {
                            "type": QUERY_INPUT_TYPE,
                            "id": key,
                        },
                        value,
                        view_mode,
                    ),
                ],
                className="mb-3",
            )
            for key, value in query_data.items()
        ]

    def get_few_shots_div_layout(self, size: int) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Pagination(
                            id="few_shots_pagination",
                            max_value=size,
                            active_page=1,
                        ),
                        dbc.Button(
                            "add example",
                            id="add_example_button",
                            outline=True,
                            size="sm",
                            color="primary",
                            className="me-1",
                        ),
                        dbc.Button(
                            "delete current example",
                            id="del_example_button",
                            outline=True,
                            size="sm",
                            color="primary",
                            className="me-1",
                        ),
                        dbc.Button(
                            "retrieve",
                            id="retrieve_button",
                            outline=True,
                            size="sm",
                            color="primary",
                            className="me-1",
                        ),
                        get_switch_layout(
                            id={
                                "type": "view_mode",
                                "id": FEW_SHOTS_INPUT,
                            },
                            labels=["view mode"],
                        ),
                    ]
                ),
                dbc.Container(id="few_shots_pagination_content"),
            ],
            id="few_shots_div",
        )

    def run(self, utils: Dict, params: Dict) -> html.Div:
        utils = {key.split(SEPARATOR_ID)[-1]: value for key, value in utils.items()}
        self.sandbox_init()
        llm = get_code_execution_model(
            **current_app.config['data_explorer']['server'],
            sandbox=self.sandbox,
        )

        logging.info(f"query to process: {params['prompts'][0]}")

        inference_cfg = get_config(InferenceConfig, utils, get_settings())

        try:
            outputs = llm.generate(
                prompts=params['prompts'],
                stop_phrases=current_app.config['data_explorer']['prompt']['stop_phrases'],
                **asdict(inference_cfg),
            )
        except requests.exceptions.ConnectionError as e:
            return self._get_connection_error_message()
        except Exception as e:
            logging.error(f"error during run prompt: {e}")
            logging.error(f"error type: {type(e)}")
            return html.Div(f"Got error\n{e}")

        logging.info(f"query's answer: {outputs[0]}")

        try:
            predicted_answer = extract_answer(outputs[0]['generation'])
            color, background, is_correct = (
                ('#d4edda', '#d4edda', "correct")
                if self.sandbox.is_output_correct(predicted_answer, params["expected_answer"])
                else ("#fecccb", "#fecccb", "incorrect")
            )
        except Exception as e:
            color, background, is_correct = 'black', 'white', "unknown"
        return html.Div(
            [
                get_results_content_layout(
                    outputs[0]['generation'],
                    get_single_prompt_output_layout(
                        outputs[0]['generation'],
                    ),
                    style={"border": f"2px solid {color}"},
                    switch_is_active=True,
                ),
                html.Div(
                    (
                        f"Answer {predicted_answer} is {is_correct}"
                        if is_correct != "unknown"
                        else "Could not evaluate the answer"
                    ),
                    style={"background-color": background},
                ),
            ]
        )

    def get_prompt(self, utils: Dict, input_dict: Dict[str, str]) -> str:
        utils = {
            key.split(SEPARATOR_ID)[-1]: value
            for key, value in utils.items()
            if key != RETRIEVAL and key not in RETRIEVAL_FIELDS
        }
        examples_type = utils.pop('examples_type', None)
        utils["example_dicts"] = get_examples().get(
            examples_type,
            [],
        )[: utils.get('num_few_shots', -1)]
        len_example_dicts = len(utils["example_dicts"])
        utils['num_few_shots'] = min(len_example_dicts, utils.get('num_few_shots', len_example_dicts))
        prompt_config = get_config(PromptConfig, utils, get_settings())

        prompt_config.few_shot_examples = get_config(
            FewShotExamplesConfig,
            utils,
            get_settings(),
        )

        prompt = Prompt(config=prompt_config)
        return prompt.build_string(input_dict)

    def _get_search_prompt_layout(self) -> dbc.InputGroup:
        return dbc.InputGroup(
            [
                dbc.InputGroupText("Index of test"),
                dbc.Input(
                    value=1,
                    id="query_search_input",
                    type="number",
                    size="sm",
                ),
                dbc.Button(
                    "Search",
                    id="query_search_button",
                    outline=True,
                    size="sm",
                    color="primary",
                    className="me-1",
                ),
            ],
            className="mb-3",
        )

    def _get_connection_error_message(self):
        return html.Div(
            html.P(
                [
                    "Could not connect to the server. Please check that the server is running (look at ",
                    html.A(
                        "inference.md",
                        href="https://github.com/Kipok/NeMo-Skills/blob/main/docs/inference.md",
                    ),
                    " for more information). ",
                    "Also check that you have provided correct host, ssh_key_path and ssh_server parameters",
                ]
            )
        )
