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

import re
from typing import Tuple

CODE_SEPARATORS = ('<llm-code>', '</llm-code>')  # used to execute code within these tags
CODE_OUTPUT_SEPARATORS = ('<llm-code-output>', '</llm-code-output>')  # used to extract the code output


def _extract_between_separators(generation, separators: Tuple[str, str], extract_all=False):
    """Extracting all text between last occurrence of separators[0] and [1].

    If extract_all is True, returning a list with all occurrences of text between separators.
    """
    if extract_all:
        pattern = f'{separators[0]}(.*?){separators[1]}'
        return re.findall(pattern, generation, re.DOTALL)
    return generation.split(separators[0])[-1].split(separators[1])[0]


def extract_code_to_execute(generation, extract_all=False):
    return _extract_between_separators(generation, CODE_SEPARATORS, extract_all)


def extract_code_output(generation, extract_all=False):
    return _extract_between_separators(generation, CODE_OUTPUT_SEPARATORS, extract_all)


def extract_error_message(generation):
    """Parsing output for any error messages and returning if found."""
    from nemo_skills.code_execution.sandbox import Sandbox

    if CODE_OUTPUT_SEPARATORS[0] not in generation:
        return Sandbox.NOT_EXECUTED
    code_output = generation.split(CODE_OUTPUT_SEPARATORS[0])[-1].split(CODE_OUTPUT_SEPARATORS[1])[0].strip()
    for prefix in Sandbox.ERROR_PREFIXES:
        if code_output.startswith(prefix):
            return code_output
    return ""
