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

import pytest
from test_code_execution import _get_sandbox


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
@pytest.mark.parametrize(
    "output_pair",
    [
        (5, 5),
        (5, 5.0),
        ("1/2", 0.5),
        ("3128 1/2", 3128.5),
        ("\\frac{1}{2}", 0.5),
        ("918\\frac{1}{2}", 918.5),
        ("x^2+2x+1", "x^2 + 2*x + 1"),
        ("x^2+2x+1", "x^2 + 2*x - (-1)"),
        ("x^2+2x+1", "2x+ 1+x^2"),
        ("odd", "\\text{odd}"),
        ("E", "\\mathrm{E}"),
        ("B", "\\mathcal{B}"),
        ("A", "\\textbf{A}"),
        ("f'", "f'"),
        ("185", "185\\"),
        ("185\\", "185\\"),
        (".185", "0.185"),
        ("\\frac {1}{2}", 0.5),
        ("17\\text{ any text}", "17"),
        ("\$10", "10"),
        ("10%", "0.1"),
        ("56,\\!01,\\!78,\\!95,\\!760", "56017895760"),
        (5 / 2, '\\frac{5}{2}'),
        ('\\frac{1}{3}', '\\dfrac{1}{3}'),
        ('(r+5)(r+5)', '(r+5)^2'),
    ],
    ids=str,
)
def test_correct_examples(sandbox_type, output_pair):
    sandbox = _get_sandbox(sandbox_type)

    output = sandbox.is_output_correct(output_pair[0], output_pair[1])
    assert output is True
    output = sandbox.is_output_correct(output_pair[1], output_pair[0])
    assert output is True


@pytest.mark.parametrize("sandbox_type", ['local', 'piston'])
@pytest.mark.parametrize(
    "output_pair",
    [
        (5, 5.001),
        (0, None),
        ("x^2+2x+1", "x^3+2x+1"),
        ("odd", "\\text{oddd}"),
        ("E", "\\mathrm{E}*2"),
        ("E", "E'"),
    ],
    ids=str,
)
def test_incorrect_examples(sandbox_type, output_pair):
    sandbox = _get_sandbox(sandbox_type)

    output = sandbox.is_output_correct(output_pair[0], output_pair[1])
    assert output is False
    output = sandbox.is_output_correct(output_pair[1], output_pair[0])
    assert output is False
