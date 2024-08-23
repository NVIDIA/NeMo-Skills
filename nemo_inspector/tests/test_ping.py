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
import subprocess
import sys
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

project_root = str(Path(__file__).parents[2])
sys.path.remove(str(Path(__file__).parents[0]))


@pytest.fixture(scope="module")
def nemo_inspector_process():
    # Start the NeMo Inspector as a subprocess

    process = subprocess.Popen(
        ["python", "nemo_inspector/nemo_inspector.py"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    yield process

    # Terminate the process after the tests
    process.terminate()
    process.wait()


@pytest.fixture
def chrome_driver():
    chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
    options = Options()
    options.page_load_strategy = 'normal'
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    os.environ['PATH'] += os.pathsep + '/'.join(chrome_driver_path.split("/")[:-1])
    yield driver
    driver.quit()


@pytest.mark.parametrize(
    ("element_id", "url"),
    [('run_button', "/"), ('add_model', "/analyze")],
)
def test_dash_app_launch(chrome_driver, nemo_inspector_process, element_id, url):
    full_url = f"http://localhost:8080{url}"

    chrome_driver.get(full_url)

    element = WebDriverWait(chrome_driver, 10).until(EC.presence_of_element_located((By.ID, element_id)))
    assert element.is_displayed()
