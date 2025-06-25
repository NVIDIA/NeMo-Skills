# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/gencode.py

import ast
import logging
import re
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def extract_function_name(function_header):
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r'\bclass\s+(\w+)\s*\('
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError('Function name or class name not found.')


def get_function_from_code(code_string, function_name):
    """
    Extracts and returns the source code of the specified function from a given source code string.

    :param code_string: String containing Python source code
    :param function_name: Name of the function to extract
    :return: String containing the source code of the function, or None if the function is not found
    """
    if code_string is None:
        return None
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
        # Iterate through all nodes in the AST
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                # Convert the AST back to a string containing the Python code for the function
                return ast.unparse(node)
    except Exception as e:
        LOG.info(f'{function_name} not found with error: {e}')
        return code_string


def process_problem_code(prob_data: dict, num_steps: int) -> str:
    header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
    return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
    string = f"{header_docstring}\n\n{return_str}"
    return string


def process_problem_steps(problem_data: dict, num_steps: int, previous_llm_code, with_background):
    """Process problem data and return previous steps and next steps"""
    output_lines = []
    next_step = []
    previous_code = []
    for i in range(num_steps - 1):
        output_lines.append(
            problem_data["sub_steps"][i]["step_description_prompt"]
            + '\n'
            + problem_data["sub_steps"][i]["step_background"]
            if with_background
            else problem_data["sub_steps"][i]["step_description_prompt"]
        )
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_step.append(
        problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
        + '\n'
        + problem_data["sub_steps"][num_steps - 1]["step_background"]
        if with_background
        else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
    )
    next_step.append(process_problem_code(problem_data, num_steps))
    output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return output_str, next_step_str, previous_code_str


def generate_prompt_with_steps(prob_data: dict, num_steps: int, prompt_template, previous_llm_code, with_background):
    # parse the input file and extract the content
    problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
        prob_data, num_steps, previous_llm_code, with_background
    )
    dependencies = prob_data["required_dependencies"]
    assert next_step_str
    return (
        prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ),
        f'{dependencies}\n{previous_code_str}\n',
    )


def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if '```python' in response
            else response.split('```')[1].split('```')[0]
        )
    else:
        LOG.info("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script


def save_response_with_steps(prob_data: dict, response: str, previous_code: str, num_steps: int, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_id = prob_data["problem_id"]
    output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
    python_code = extract_python_script(response)
    output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")


def generate_response_with_steps(
    prob_data: dict, num_steps: int, tot_steps: int, prompt_template, previous_llm_code, with_background
) -> None:
    prob_id = prob_data["problem_id"]
    for prev_step in range(num_steps - 1):
        if previous_llm_code[prev_step] is None:
            if (
                (prob_id == "13" and prev_step == 5)
                or (prob_id == "62" and prev_step == 0)
                or (prob_id == "76" and prev_step == 2)
            ):
                # TODO
                prev_file_path = Path("/workspace/SciCode/eval", "data", f"{prob_id}.{prev_step+1}.txt")
            else:
                prev_file_path = Path('./tmp-scicode-dir' / f"{prob_id}.{prev_step + 1}.py")
            if prev_file_path.is_file():
                prev_file_content = prev_file_path.read_text(encoding='utf-8')
                func_name = extract_function_name(prob_data["sub_steps"][prev_step]["function_header"])
                function_code = get_function_from_code(prev_file_content, func_name)
                previous_llm_code[prev_step] = function_code
            else:
                raise Exception(f'Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}.')

    prompt, previous_code = generate_prompt_with_steps(
        prob_data, num_steps, prompt_template, previous_llm_code, with_background
    )

    # model_kwargs = {}
    # if "claude" in model:
    #     model_kwargs["max_tokens"] = 4096
    # model_kwargs["temperature"] = self.temperature
    # # write the response to a file if it doesn't exist
    # model_fct = get_model_function(model, **model_kwargs)
    response_from_llm = "here is my response"  # model_fct(prompt)
    previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
    save_response_with_steps(prob_data, response_from_llm, previous_code, num_steps, Path('./tmp-scicode-dir'))
    return previous_llm_code
