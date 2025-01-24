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
import re

import pytest

from nemo_skills.code_execution import extract_code_output, extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.prompt.few_shot_examples import examples_map


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
    print(output)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': '1\n'}
    assert session_id is not None

    code = "a + 5"
    output, session_id2 = sandbox.execute_code(code, session_id=session_id)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': '6\n'}
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
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': 'asdf\n'}
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_multiple_prints(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """
    print("1")
    print("2x3")
    """

    output, session_id = sandbox.execute_code(code)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': '1\n2x3\n'}
    assert session_id is not None

    code = "print(2)\n15"
    output, session_id2 = sandbox.execute_code(code, session_id=session_id)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': '2\n15\n'}
    assert session_id == session_id2


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_no_output(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """a = 2"""

    output, session_id = sandbox.execute_code(code)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': ''}
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_execution_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """1 / 0"""

    output, session_id = sandbox.execute_code(code)
    # TODO: somehow in our current implementation errors also go to stdout. How to fix this?
    error = (
        '\x1b[0;31m---------------------------------------------------------------------------\x1b[0m\n\x1b[0;31m'
        'ZeroDivisionError\x1b[0m                         Traceback (most recent call last)\nFile \x1b[0;32m'
        '<ipython-input-1-bc757c3fda29>:1\x1b[0m\n\x1b[0;32m----> 1\x1b[0m \x1b[38;5;241;43m1\x1b[39;49m\x1b[43m '
        '\x1b[49m\x1b[38;5;241;43m/\x1b[39;49m\x1b[43m \x1b[49m\x1b[38;5;241;43m0\x1b[39;49m\n\n\x1b[0;31m'
        'ZeroDivisionError\x1b[0m: division by zero\n'
    )
    assert output == {
        'process_status': 'completed',
        'stderr': '',
        'stdout': error,
    }
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_syntax_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """a = 2\n b = 3"""

    output, session_id = sandbox.execute_code(code)
    error = (
        '\x1b[0;36m  File \x1b[0;32m<ipython-input-1-ff73a4eb1351>:2\x1b[0;36m\x1b[0m\n\x1b[0;31m    '
        'b = 3\x1b[0m\n\x1b[0m    ^\x1b[0m\n\x1b[0;31mIndentationError\x1b[0m\x1b[0;31m:\x1b[0m unexpected indent\n'
    )
    assert output == {
        'process_status': 'completed',
        'stderr': '',
        'stdout': error,
    }
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_timeout_error(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    code = """import time\ntime.sleep(1)\nprint("done")"""

    output, session_id = sandbox.execute_code(code, timeout=1)
    assert output == {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
    assert session_id is not None

    output, session_id = sandbox.execute_code(code, timeout=2, session_id=session_id)
    assert output == {'process_status': 'completed', 'stderr': '', 'stdout': 'done\n'}
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
    error = (
        "\x1b[0;31m---------------------------------------------------------------------------\x1b[0m\n\x1b[0;31m"
        "NameError\x1b[0m                                 Traceback (most recent call last)\nFile \x1b[0;32m"
        "<ipython-input-1-2d264478936f>:4\x1b[0m\n\x1b[1;32m      2\x1b[0m height_in_inches \x1b[38;5;241m=\x1b"
        "[39m \x1b[38;5;241m20\x1b[39m \x1b[38;5;241m*\x1b[39m \x1b[38;5;241m12\x1b[39m\n\x1b[1;32m      3\x1b[0m"
        " \x1b[38;5;66;03m# height of bamboo in inches after x days\x1b[39;00m\n\x1b[0;32m----> 4\x1b[0m "
        "height_after_x_days \x1b[38;5;241m=\x1b[39m height_in_inches \x1b[38;5;241m+\x1b[39m \x1b[38;5;241m30"
        "\x1b[39m \x1b[38;5;241m*\x1b[39m \x1b[43mx\x1b[49m\n\x1b[1;32m      5\x1b[0m \x1b[38;5;66;03m"
        "# solve for x\x1b[39;00m\n\x1b[1;32m      6\x1b[0m x \x1b[38;5;241m=\x1b[39m (\x1b[38;5;241m600\x1b[39m "
        "\x1b[38;5;241m-\x1b[39m height_in_inches) \x1b[38;5;241m/\x1b[39m \x1b[38;5;241m30\x1b[39m\n\n\x1b[0;31m"
        "NameError\x1b[0m: name 'x' is not defined\n"
    )
    output, session_id = sandbox.execute_code(code)
    assert output == {
        'process_status': 'completed',
        'stderr': '',
        'stdout': error,
    }
    assert session_id is not None


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
@pytest.mark.parametrize(
    "code_begin,code_end,code_output_begin,code_output_end,code_output_format",
    [
        ('<llm-code>\n', '</llm-code>\n', '<llm-code-output>\n', '</llm-code-output>\n', 'qwen'),
        (
            '<|python_tag|>',
            '<|eom_id|>',
            '<|start_header_id|>ipython<|end_header_id|>',
            '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
            'llama',
        ),
    ],
)
def test_few_shots(sandbox_type, code_begin, code_end, code_output_begin, code_output_end, code_output_format):
    def replace_code_output(match):
        code_output = match.group(2)
        formatted_output = format_code_output(
            execution_dict={"process_status": "completed", "stdout": code_output, "stderr": ""},
            code_output_begin=code_output_begin,
            code_output_end=code_output_end,
            code_output_format=code_output_format,
        )
        return formatted_output

    sandbox = _get_sandbox(sandbox_type)

    for example_name, example_list in examples_map.items():
        for example in example_list:
            if 'solution' not in example:
                continue
            example = example.copy()

            pattern = r'({code_output_begin}\n)(.*?)({code_output_end})'
            example["solution"] = re.sub(pattern, replace_code_output, example["solution"], flags=re.DOTALL)
            example["solution"] = example["solution"].replace("{code_begin}", code_begin)
            example["solution"] = example["solution"].replace("{code_end}", code_end)
            example["solution"] = example["solution"].replace("{code_output_begin}", "")
            example["solution"] = example["solution"].replace("{code_output_end}", "")

            if len(extract_code_to_execute(example['solution'], code_begin, code_end, extract_all=True)) > 0:
                code_snippets = extract_code_to_execute(example['solution'], code_begin, code_end, extract_all=True)
                if code_output_format == 'qwen':
                    expected_outputs = extract_code_output(
                        example['solution'], code_output_begin, code_output_end, extract_all=True
                    )
                    expected_outputs = [(output, None) for output in expected_outputs]
                elif code_output_format == 'llama':
                    pattern = r'\[stdout\]\n(.*?)\[/stdout\]|\[stderr\]\n(.*?)\[/stderr\]'
                    expected_outputs = re.findall(pattern, example['solution'], re.DOTALL)
                session_id = None
                for code_snippet, (expected_output, expected_error) in zip(code_snippets, expected_outputs):
                    if not expected_error:
                        expected_error = None
                    output, session_id = sandbox.execute_code(code_snippet, session_id=session_id)
                    execution_dict = {
                        "process_status": "completed",
                        "stdout": expected_output,
                        "stderr": expected_error,
                    }
                    generated_output = format_code_output(
                        output, code_output_begin, code_output_end, code_output_format
                    )
                    extracted_output = format_code_output(
                        execution_dict, code_output_begin, code_output_end, code_output_format
                    )
                    assert generated_output == extracted_output, f"{example_name} few shots are failing"


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_lean4_basic_code_execution(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    # Test case for correct basic Lean4 code execution
    correct_code = """
    -- Test.lean
    def add (a b : Nat) : Nat :=
      a + b

    #eval add 3 4  -- This should print 7
    """
    expected_output = "7\n"

    output, session_id = sandbox.execute_code(correct_code, language="lean4")

    # Assertions for the correct code
    assert session_id == None
    assert output["process_status"] == 'completed', "Expected the process to complete successfully"
    assert expected_output == output["stdout"], f"Expected the output to include '{expected_output}'"
    assert output["stderr"] == "", "Expected no error output"


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_lean4_mathlib_code_execution(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    # Test case for Lean4 code that imports mathlib
    correct_code_mathlib = """
    -- Test_mathlib.lean
    import Mathlib
    #eval 7
    """
    expected_output = "7\n"

    output, session_id = sandbox.execute_code(correct_code_mathlib, language="lean4")

    # Assertions for the mathlib code
    assert session_id == None
    assert output["process_status"] == 'completed', "Expected the process to complete successfully"
    assert expected_output == output["stdout"], f"Expected the output to include '{expected_output}'"
    assert output["stderr"] == "", "Expected no error output"


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_lean4_code_execution_failure(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    # Test case for Lean4 code with syntax error
    incorrect_code = """
    -- Test_fail.lean
    def add (a b : Nat) : Nat :=
      a +  -- Syntax error here

    #eval add 3 4
    """

    error_output, session_id = sandbox.execute_code(incorrect_code, language="lean4")

    # Assertions for the error case
    assert session_id == None
    print(error_output)
    assert error_output["process_status"] == 'failed', "Expected the process to fail due to syntax error"
    assert (
        "unexpected token '#eval" in error_output["stdout"].lower()
    ), "Expected the error output to mention an unexpected token '#eval"


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
def test_lean4_few_shots(sandbox_type):
    sandbox = _get_sandbox(sandbox_type)

    from nemo_skills.prompt.few_shot_examples.minif2f import minif2f_deepseek_fewshot

    # Test case for Lean4 code with syntax error
    session_id_list = []
    process_status_list = []
    stdout_list = []
    stderr_list = []

    for i, entry in enumerate(minif2f_deepseek_fewshot):
        code = entry["header"] + entry["informal_prefix"] + entry["formal_statement"] + entry["formal_proof"]

        output, session_id = sandbox.execute_code(code, language="lean4")

        if session_id is not None:
            session_id_list.append(i)
        if output["process_status"] != 'completed':
            process_status_list.append(i)
        if output["stdout"] != "":
            stdout_list.append(i)
        if output["stderr"] != "":
            stderr_list.append(i)

    # Assertions for the correct code
    assert (
        not session_id_list
    ), f"Expected session_id to be None for all test cases, but got session_ids for few shots at indices {session_id_list}."
    assert (
        not process_status_list
    ), f"Expected process_status to be 'completed' for all test cases, but these few shots did not complete successfully: indices {process_status_list}."
    assert (
        not stdout_list
    ), f"Expected the stdout to match the expected output for all test cases, but mismatches were found at indices {stdout_list}."
    assert (
        not stderr_list
    ), f"Expected no errors in stderr for all test cases, but errors were found at indices {stderr_list}."
