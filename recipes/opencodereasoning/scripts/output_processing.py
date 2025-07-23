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

import ast


def check_generation(response, keep_explanations=False, do_ast_check=False) -> bool:
    """Obtained from https://github.com/nickrosh/evol-teacher/blob/main/generate_evol.py

    Check the generated instruction with several checks. If it returns True,
    then the instruction should be discarded

    Reviews the generation from a language model and checks if it is valid.

    Args:
        response: The response to be checked.
        keep_explanations: Whether to keep explanations in the generation.
        do_ast_check: Whether to check the generation with AST.

    Results:
        bool: Whether the instruction should be discarded.
    """

    # If hit max seq length, ignore sample
    if response["finish_reason"] == "length":
        return True

    content = response["text"]  # type: str

    # print("Content: ", content)

    # Check if the content is empty
    if not content:
        return True

    # Check if the content is too short
    if len(content.split()) <= 3:
        return True

    # Check if the content is non-ascii
    if not content[0].isascii():
        return True

    # Some common cases where the model tries to cheat by providing just a placeholder text
    if "your code here" in content.lower():
        return True

    # Case where Command codes leaks into the generation
    if "#instruction#" in content.lower() or "#output#" in content.lower():
        return True
    # if "write" in content.lower() and ("code" in content.lower() or "program" in content.lower()):
    #     return True

    # Check if the content is too long
    # if response["total_tokens"] >= 1000:
    #     return True

    # Cache original content
    response['original_text'] = response['text']
    original_content = content

    # Replace multiple space lines with tab
    content = content.replace('    ', '\t')

    # Remove explanation with some heuristics
    if "explain" in content.lower():
        def_index = content.find("def ")
        explain_index = content.lower().find("explain")

        # If def comes before explain, then remove the explain part
        if def_index != -1 and explain_index != -1 and def_index < explain_index:
            response['text'] = content[:explain_index]
            content = response['text']

    # Extract code block(s)
    content_header = None
    if '```' in content:
        # Find the begin and end code block
        first_index = content.find('```')
        second_index = content.find('```', first_index + 1)

        # Find the first newline after the first code block
        first_index_newline = content.find('\n', first_index + 3)

        # If both code blocks are found and the first one comes before the second one
        if first_index != -1 and second_index != -1 and first_index < second_index:
            # Extract the content header and the text content from the code block
            # If the first newline comes before the second code block, then extract the content header
            if first_index_newline != -1 and first_index_newline < second_index:
                content_header = content[first_index : first_index_newline + 1]
                response['text'] = content[first_index_newline + 1 : second_index].strip()
            else:
                # Extract the content header and the text content from the code block
                content_header = content[first_index : first_index + len('```')]
                response['text'] = content[first_index + len('```') : second_index].strip()

            # If the response starts with python, then remove it (due to removal of ``` from ```python)
            if response['text'].startswith('python'):
                response['text'] = response['text'][len('python') :].strip()

            content = response['text']

    if "pass" in content:
        return True

    # Do AST check only if the language is python
    if do_ast_check and "def" in content:
        try:
            ast.parse(content)
        except Exception as e:
            return True

    # If keep_explanations is True, then update the content and response['text'] to original content
    if keep_explanations:
        # Update content and response['text'] to original content
        content = original_content
        response['text'] = original_content

    # If content header is not None and keep_explanations is False, then update the response['text'] to include the
    if content_header is not None and not keep_explanations:
        content = content_header + response['text'] + '\n```'
        response['text'] = content_header + response['text'] + '\n```'

    return False


def post_process_generation(response, keep_explanations=False, do_ast_check=False):
    """
    Post process the generation instructions by filtering out the unwanted instructions.

    Args:
        response: The response to be post processed.
        keep_explanations: Whether to keep explanations in the generation.
        do_ast_check: Whether to check the generation with AST.

    Returns:
        instructions: The filtered instructions or None if the instruction is invalid.
    """
    if response is None:
        return None

    # Check if the instruction is valid
    if check_generation(response, keep_explanations=keep_explanations, do_ast_check=do_ast_check):
        return None
    else:
        return response
