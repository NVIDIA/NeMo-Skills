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
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[1]))


# Mocking the set_config function
def mock_set_config():
    pass


@pytest.fixture
def dash_app():
    with patch('hydra.main', lambda *args, **kwargs: lambda func: mock_set_config):
        from nemo_inspector.layouts import get_main_page_layout
        from nemo_inspector.nemo_inspector import app

        app.title = "NeMo Inspector"
        app.layout = get_main_page_layout()
        config = {
            'nemo_inspector': {
                'prompt': {
                    'prompt_type': '',
                    'context_template': '',
                    'context_type': '',
                    'few_shot_examples': {'examples_type': '', 'num_few_shots': 0},
                },
                'types': {
                    'prompt_type': [],
                    'examples_type': [],
                    'context_type': [],
                    'retrieval_field': [''],
                },
                'data_file': 'mock_data_file',
                'inspector_params': {
                    'model_prediction': {},
                    'save_generations_path': 'mock_save_generations_path',
                },
                'server': {},
                'sandbox': {},
            }
        }
        server = Flask(__name__)
        server.config.update(config)
        app.server.config.update(config)

        return app


@pytest.fixture
def chrome_driver():
    chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
    options = Options()
    options.page_load_strategy = 'normal'
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if 'GITHUB_ACTIONS' in os.environ:
        # Running in GitHub Actions
        driver = webdriver.Chrome(options=options)
    else:
        # Running locally
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

    os.environ['PATH'] += os.pathsep + '/'.join(chrome_driver_path.split("/")[:-1])
    yield driver
    driver.quit()


@pytest.mark.parametrize(
    ("element_id", "url"),
    [('run_button', "/"), ('add_model', "/analyze")],
)
def test_dash_app_launch(chrome_driver, dash_duo, dash_app, element_id, url):
    dash_duo.start_server(dash_app)

    server_url = dash_duo.server_url
    full_url = f"{server_url}{url}"

    chrome_driver.get(full_url)

    try:
        element = WebDriverWait(chrome_driver, 10).until(EC.presence_of_element_located((By.ID, element_id)))
        assert element.is_displayed()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Page source: {chrome_driver.page_source}")
        raise
