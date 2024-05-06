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

import pytest

from nemo_skills.code_execution import extract_code_output, extract_code_to_execute
from nemo_skills.code_execution.sandbox import Sandbox, get_sandbox
from nemo_skills.inference.prompt.few_shot_examples import examples_map


def _get_sandbox(sandbox_type):
    if sandbox_type == 'local':
        host = os.getenv('NEMO_SKILLS_SANDBOX_HOST')
        if not host:
            pytest.skip("Define NEMO_SKILLS_SANDBOX_HOST to run this test")

    if sandbox_type == 'piston':
        host = os.getenv('NEMO_SKILLS_PISTON_SANDBOX_URL')
        if not host:
            pytest.skip("Define NEMO_SKILLS_PISTON_SANDBOX_URL to run this test")

    return get_sandbox(sandbox_type, host=host)


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_multiple_code_blocks(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """
    a = 1
    a
    """

    output, session_id = sandbox.execute_code(code)
    assert output == {'result': '1', 'error_message': ''}
    assert session_id is not None

    code = "a + 5"
    output, session_id2 = sandbox.execute_code(code, session_id=session_id)
    assert output == {'result': '6', 'error_message': ''}
    assert session_id == session_id2


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_triple_quotes(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)
    code = '''
    def my_func():
        """Test function"""
        print("asdf")
    my_func()
'''
    output, session_id = sandbox.execute_code(code)
    assert output == {'result': 'asdf', 'error_message': ''}
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_multiple_prints(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """
    print("1")
    print("2x3")
    """

    output, session_id = sandbox.execute_code(code)
    assert output == {'result': '1\n2x3', 'error_message': ''}
    assert session_id is not None

    code = "print(2)\n15"
    output, session_id2 = sandbox.execute_code(code, session_id=session_id)
    assert output == {'result': '2\n15', 'error_message': ''}
    assert session_id == session_id2


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_no_output(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """a = 2"""

    output, session_id = sandbox.execute_code(code)
    assert output == {'result': '', 'error_message': Sandbox.RESULT_NOT_DEFINED_ERROR}
    assert session_id is None  # we are clearing the sessions on error, so it should be None here


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_execution_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """1 / 0"""

    output, session_id = sandbox.execute_code(code)
    assert output == {
        'result': (
            '\x1b[0;31m---------------------------------------------------------------------------\x1b[0m\n\x1b[0;31m'
            'ZeroDivisionError\x1b[0m                         Traceback (most recent call last)\nFile \x1b[0;32m'
            '<ipython-input-1-bc757c3fda29>:1\x1b[0m\n\x1b[0;32m----> 1\x1b[0m \x1b[38;5;241;43m1\x1b[39;49m\x1b[43m '
            '\x1b[49m\x1b[38;5;241;43m/\x1b[39;49m\x1b[43m \x1b[49m\x1b[38;5;241;43m0\x1b[39;49m\n\n\x1b[0;31m'
            'ZeroDivisionError\x1b[0m: division by zero'
        ),
        'error_message': f'{Sandbox.EXECUTION_ERROR} division by zero',
    }
    assert session_id is None  # we are clearing the sessions on error, so it should be None here


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_syntax_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """a = 2\n b = 3"""

    output, session_id = sandbox.execute_code(code)
    assert output == {
        'result': (
            '\x1b[0;36m  File \x1b[0;32m<ipython-input-1-ff73a4eb1351>:2\x1b[0;36m\x1b[0m\n\x1b[0;31m    '
            'b = 3\x1b[0m\n\x1b[0m    ^\x1b[0m\n\x1b[0;31mIndentationError\x1b[0m\x1b[0;31m:\x1b[0m unexpected indent'
        ),
        'error_message': f'{Sandbox.SYNTAX_ERROR} unexpected indent (<ipython-input-1-ff73a4eb1351>, line 2)',
    }
    assert session_id is None  # we are clearing the sessions on error, so it should be None here


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_timeout_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """import time\ntime.sleep(1)\nprint("done")"""

    output, session_id = sandbox.execute_code(code, timeout=1)
    assert output == {
        'result': None,
        'error_message': Sandbox.TIMEOUT_ERROR,
    }
    assert session_id is None  # we are clearing the sessions on error, so it should be None here

    output, session_id = sandbox.execute_code(code, timeout=2, session_id=session_id)
    assert output == {
        'result': "done",
        'error_message': "",
    }
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_real_generations(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """
# height of bamboo in inches
height_in_inches = 20 * 12
# height of bamboo in inches after x days
height_after_x_days = height_in_inches + 30 * x
# solve for x
x = (600 - height_in_inches) / 30
x
"""

    output, session_id = sandbox.execute_code(code)
    assert output == {
        'result': (
            "\x1b[0;31m---------------------------------------------------------------------------\x1b[0m\n\x1b[0;31m"
            "NameError\x1b[0m                                 Traceback (most recent call last)\nFile \x1b[0;32m"
            "<ipython-input-1-2d264478936f>:4\x1b[0m\n\x1b[1;32m      2\x1b[0m height_in_inches \x1b[38;5;241m=\x1b"
            "[39m \x1b[38;5;241m20\x1b[39m \x1b[38;5;241m*\x1b[39m \x1b[38;5;241m12\x1b[39m\n\x1b[1;32m      3\x1b[0m"
            " \x1b[38;5;66;03m# height of bamboo in inches after x days\x1b[39;00m\n\x1b[0;32m----> 4\x1b[0m "
            "height_after_x_days \x1b[38;5;241m=\x1b[39m height_in_inches \x1b[38;5;241m+\x1b[39m \x1b[38;5;241m30"
            "\x1b[39m \x1b[38;5;241m*\x1b[39m \x1b[43mx\x1b[49m\n\x1b[1;32m      5\x1b[0m \x1b[38;5;66;03m"
            "# solve for x\x1b[39;00m\n\x1b[1;32m      6\x1b[0m x \x1b[38;5;241m=\x1b[39m (\x1b[38;5;241m600\x1b[39m "
            "\x1b[38;5;241m-\x1b[39m height_in_inches) \x1b[38;5;241m/\x1b[39m \x1b[38;5;241m30\x1b[39m\n\n\x1b[0;31m"
            "NameError\x1b[0m: name 'x' is not defined"
        ),
        'error_message': f"{Sandbox.EXECUTION_ERROR} name 'x' is not defined",
    }
    assert session_id is None  # we are clearing the sessions on error, so it should be None here


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_few_shots(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    for example_name, example_list in examples_map.items():
        for example in example_list:
            if len(extract_code_to_execute(example['generated_solution'], extract_all=True)) > 1:
                code_snippets = extract_code_to_execute(example['generated_solution'], extract_all=True)
                expected_outputs = extract_code_output(example['generated_solution'], extract_all=True)
                session_id = None
                for code_snippet, expected_output in zip(code_snippets, expected_outputs):
                    output, session_id = sandbox.execute_code(code_snippet, session_id=session_id)
                    assert output['result'] == expected_output.strip(), f"{example_name} few shots are failing"
