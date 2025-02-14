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
import re
from typing import Dict, Tuple

LOG = logging.getLogger(__name__)


def format_code_output(
    execution_dict: Dict[str, str], code_output_begin: str, code_output_end: str, code_output_format: str = 'llama'
):
    """Formatting code output to be displayed as an llm expects it."""
    if code_output_format == 'llama':
        output = execution_dict["process_status"]
        if execution_dict['stdout']:
            output += f"\n[stdout]\n{execution_dict['stdout']}[/stdout]"
        if execution_dict['stderr']:
            output += f"\n[stderr]\n{execution_dict['stderr']}[/stderr]"
        output = f"{code_output_begin}\n\n{output}{code_output_end}\n\n"
    elif code_output_format == 'qwen':
        output = ""
        if execution_dict['stdout']:
            output += f"{execution_dict['stdout']}"
        if execution_dict['stderr']:
            output += f"{execution_dict['stderr']}"
        if execution_dict['stderr'] and execution_dict['stdout']:
            LOG.warning("Both stdout and stderr are not empty. This shouldn't normally happen! %s", execution_dict)
        output = f"{code_output_begin}{output}{code_output_end}"
    else:
        raise ValueError(f"Unknown code_output_format: {code_output_format}")

    # wrapping with code output separators
    return output


def _extract_between_separators(generation: str, separators: Tuple[str, str], extract_all: bool = False):
    """Extracting all text between last occurrence of separators[0] and [1].

    If extract_all is True, returning a list with all occurrences of text between separators.
    """
    if extract_all:
        separators = [re.escape(sp) for sp in separators]
        pattern = f'{separators[0]}(.*?){separators[1]}'
        return re.findall(pattern, generation, re.DOTALL)
    return generation.split(separators[0])[-1].split(separators[1])[0]


def extract_code_to_execute(generation: str, code_begin: str, code_end: str, extract_all: bool = False):
    return _extract_between_separators(generation, [code_begin, code_end], extract_all)


def extract_code_output(generation: str, code_output_begin: str, code_output_end: str, extract_all: bool = False):
    return _extract_between_separators(generation, [code_output_begin, code_output_end], extract_all)

def extract_code_block(text: str, languages: list = ["lean4", "lean3", "lean", ""]) -> str:
    for language in languages:
        match = re.search(rf"```{language}\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""

def clean_formal_generation(generation: str, languages: list = ["lean4", "lean3", "lean", ""]) -> str:
    # Extract part after **FINAL ANSWER** if present
    if "**FINAL ANSWER**" in generation:
        generation = generation.split("**FINAL ANSWER**", 1)[1].strip()
    
    extracted_code = extract_code_block(generation, languages)
    if extracted_code:
        return extracted_code
    
    # If no explicit code block, remove any surrounding triple backticks
    return re.sub(r"^\s*```(?:lean4|lean3|lean)?\s*|\s*```[\s]*$", "", generation).strip()

