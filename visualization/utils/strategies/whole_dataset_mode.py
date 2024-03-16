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
import re
from typing import Dict, List

import dash_bootstrap_components as dbc
import pandas as pd
import requests
from dash import dash_table, html
from flask import current_app
from omegaconf import OmegaConf
from settings.constants import GREEDY, OUTPUT, OUTPUT_PATH, PARAMETERS_FILE_NAME, SEPARATOR_ID, WHOLE_DATASET_MODE
from settings.templates import summarize_results_template
from utils.common import get_available_models, get_examples, run_subprocess
from utils.strategies.base_strategy import ModeStrategies

from nemo_skills.evaluation.evaluate_results import EvaluateResultsConfig, evaluate_results
from nemo_skills.inference.generate_solutions import GenerateSolutionsConfig, InferenceConfig, generate_solutions
from nemo_skills.inference.prompt.utils import FewShotExamples, PromptConfig


class WholeDatasetModeStrategy(ModeStrategies):
    mode = WHOLE_DATASET_MODE

    def __init__(self):
        super().__init__()

    def get_query_input_layout(self, dataset) -> List[dbc.AccordionItem]:
        return []

    def run(self, utils: Dict, params: Dict) -> html.Div:
        utils = {key.split(SEPARATOR_ID)[-1]: value for key, value in utils.items()}
        self.sandbox_init()
        runs_storage = get_available_models()
        results_path = current_app.config['data_explorer']['visualization_params']['results_path']
        run_index = len(runs_storage)
        metrics_directory = os.path.join(
            results_path,
            str(run_index),
        )
        random_seed_start = utils['start_random_seed'] if params['range_random_mode'] else utils['random_seed']
        random_seed_end = utils['end_random_seed'] if params['range_random_mode'] else utils['random_seed'] + 1

        generate_solutions_config = self._get_config(
            GenerateSolutionsConfig,
            utils,
            current_app.config['data_explorer'],
            {
                "output_file": "dummy",  # will be set up later
                "example_dicts": get_examples().get(
                    utils['examples_type'],
                    [],
                ),
            },
        )

        generate_solutions_config.prompt = self._get_config(
            PromptConfig,
            utils,
            current_app.config['data_explorer']['prompt'],
        )

        generate_solutions_config.prompt.few_shot_examples = self._get_config(
            FewShotExamples,
            utils,
            current_app.config['data_explorer']['prompt']['few_shot_examples'],
        )

        generate_solutions_config.inference = self._get_config(
            InferenceConfig,
            utils,
            current_app.config['data_explorer']['inference'],
        )

        for random_seed in range(random_seed_start, random_seed_end):
            file_name = GREEDY if not params['range_random_mode'] else "rs" + str(random_seed)
            output_file = os.path.join(
                metrics_directory,
                OUTPUT_PATH.format(
                    OUTPUT,
                    file_name,
                ),
            )
            generate_solutions_config.output_file = output_file
            generate_solutions_config.inference.random_seed = random_seed
            try:
                logging.info("Generate solutions")
                generate_solutions(OmegaConf.structured(generate_solutions_config))
                evaluate_results_config = EvaluateResultsConfig(
                    prediction_jsonl_files=output_file,
                    sandbox=current_app.config['data_explorer']['sandbox'],
                )
                logging.info("Evaluate results")
                evaluate_results(OmegaConf.structured(evaluate_results_config))

            except requests.exceptions.ConnectionError as e:
                return self._get_connection_error_message()
            except Exception as e:
                return html.Pre(f"Something went wrong\n{e}")

        logging.info("Summarize results")
        summarize_results = summarize_results_template.format(
            results_path=results_path,
            benchmarks=str(run_index),
        )

        _, errors, success = run_subprocess(summarize_results)
        if not success:
            return html.Pre(f"Something went wrong\n{errors}")

        runs_storage[run_index] = {
            "utils": utils,
            "examples": get_examples().get(utils["examples_type"], []),
        }

        with open(PARAMETERS_FILE_NAME, "w") as f:
            f.write(json.dumps(runs_storage))

        df = pd.read_csv(os.path.join(results_path, "results.csv"))
        return html.Div(
            [
                html.Div(
                    [
                        html.Pre(f'Done. Results are in folder\n{"/".join(output_file.split("/")[:-1])}'),
                        dash_table.DataTable(
                            id='table',
                            columns=[{"name": i, "id": i} for i in df.columns],
                            data=df.to_dict('records'),
                            style_table={'overflowX': 'auto'},
                        ),
                    ]
                ),
            ]
        )

    def get_prompt(self, utils: Dict, input_dict: Dict[str, str]) -> str:
        utils = {key.split(SEPARATOR_ID)[-1]: value for key, value in utils.items()}
        pattern = r'\{([^}]*)\}'
        keys = []
        for value in utils.values():
            if isinstance(value, str):
                keys.extend(re.findall(pattern, value))
        keys = filter(lambda x: x not in ['examples', 'context'], keys)
        input_dict = {**{key: f"***your {key}***" for key in keys}}
        return super().get_prompt(utils, input_dict)
