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
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from dash import Dash
import dash_bootstrap_components as dbc
from flask import Flask
import hydra
from omegaconf import MISSING, OmegaConf, DictConfig

from visualization.settings.visualization_config import VisualizationConfig
from settings.constants import UNDEFINED

from nemo_skills.inference.prompt.few_shot_examples import examples_map
from nemo_skills.inference.prompt.utils import (
    context_templates,
    get_prompt_config,
)

config_path = os.path.join(os.path.abspath(Path(__file__).parents[1]), "settings")

config = {}


@hydra.main(
    version_base=None, config_path=config_path, config_name="visualization_config"
)
def set_config(cfg: VisualizationConfig) -> None:
    global config

    prompt_type = UNDEFINED  # TODO detect prompt_type

    prompt_types = [
        os.path.splitext(file)[0]
        for file in os.listdir(
            Path.joinpath(
                Path(__file__).parents[2].absolute(),
                "nemo_skills/inference/prompt",
            )
        )
        if os.path.splitext(file)[1] == '.yaml'
    ]

    for name in prompt_types:
        if get_prompt_config(name) == cfg.prompt:
            prompt_type = name
            break

    if not cfg.data_file and not cfg.dataset and not cfg.split_name:
        cfg.data_file = UNDEFINED

    cfg.output_file = UNDEFINED

    def set_undefined(dict_cfg: Dict, cfg: DictConfig):
        for key, value in dict_cfg.items():
            if isinstance(value, Dict):
                setattr(cfg, key, set_undefined(value, getattr(cfg, key)))
            if value == MISSING:
                setattr(cfg, key, UNDEFINED)
        return cfg

    cfg.prompt = set_undefined(OmegaConf.to_container(cfg.prompt), cfg.prompt)

    config['data_explorer'] = asdict(OmegaConf.to_object(cfg))

    for param in ['host', 'ssh_server', 'ssh_key_path']:
        if (
            param not in config['data_explorer']['sandbox']
            and param in config['data_explorer']['server']
        ):
            config['data_explorer']['sandbox'][param] = config['data_explorer']['server'][
                param
            ]

    config['data_explorer']['prompt']['prompt_type'] = prompt_type

    config['data_explorer']['prompt']['context_template'] = context_templates[
        config['data_explorer']["prompt"]["context_type"]
    ]

    config['data_explorer']['types'] = {
        "prompt_type": [UNDEFINED] + prompt_types,
        "examples_type": [UNDEFINED] + list(examples_map.keys()),
        "context_type": [UNDEFINED] + list(context_templates.keys()),
    }

    config['data_explorer']['data_file'] = str(config['data_explorer']['data_file'])
    # All parameters in config can be modified through the application except Server and Sandbox configs
    # Following parameters are not used and should not be modified
    config['data_explorer'].pop('output_file')
    config['data_explorer'].pop('dataset')
    config['data_explorer'].pop('split_name')
    config['data_explorer'].pop('example_dicts')


set_config()
server = Flask(__name__)
server.config.update(config)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=server,
)

from callbacks.base_callback import nav_click
from callbacks.run_prompt_callbacks import add_example
from callbacks.analyze_callbacks import choose_base_model
