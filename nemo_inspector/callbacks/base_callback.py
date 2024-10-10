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
from datetime import datetime
from typing import Tuple

from callbacks import app
from dash import html
from dash.dependencies import Input, Output
from flask import current_app
from layouts import get_compare_test_layout, get_run_test_layout
from settings.constants import CODE_BEGIN, CODE_END, CODE_OUTPUT_BEGIN, CODE_OUTPUT_END
from utils.common import get_available_models, get_data_from_files, get_height_adjustment


@app.callback(
    [
        Output("page_content", "children"),
        Output("run_mode_link", "active"),
        Output("analyze_link", "active"),
    ],
    Input("url", "pathname"),
)
def nav_click(url: str) -> Tuple[html.Div, bool, bool]:
    if url == "/":
        return get_run_test_layout(), True, False
    elif url == "/analyze":
        config = current_app.config['nemo_inspector']
        config['inspector_params']['code_separators'] = (
            config['prompt']['template'][CODE_BEGIN],
            config['prompt']['template'][CODE_END],
        )
        config['inspector_params']['code_output_separators'] = (
            config['prompt']['template'][CODE_OUTPUT_BEGIN],
            config['prompt']['template'][CODE_OUTPUT_END],
        )
        get_data_from_files(datetime.now())
        get_available_models(datetime.now())
        return get_compare_test_layout(), False, True


@app.callback(
    Output("js_container", "children", allow_duplicate=True),
    [
        Input("page_content", "children"),
        Input("js_trigger", "children"),
    ],
    prevent_initial_call=True,
)
def adjust_text_area_height(content: html.Div, trigger: str) -> html.Iframe:
    return get_height_adjustment()
