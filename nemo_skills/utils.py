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

import glob
import inspect
import io
import logging
import sys
import tokenize
import typing
from dataclasses import MISSING, fields, is_dataclass


def unroll_files(prediction_jsonl_files):
    for file_pattern in prediction_jsonl_files:
        for file in sorted(glob.glob(file_pattern, recursive=True)):
            yield file


def setup_logging(disable_hydra_logs: bool = True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if disable_hydra_logs:
        # hacking the arguments to always disable hydra's output
        sys.argv.extend(
            ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none", "hydra/hydra_logging=none"]
        )


def extract_comments(code: str):
    """Extract a list of comments from the given Python code."""
    comments = []
    tokens = tokenize.tokenize(io.BytesIO(code.encode()).readline)

    for token, line, *_ in tokens:
        if token is tokenize.COMMENT:
            comments.append(line.lstrip('#').strip())

    return comments


def type_to_str(type_hint):
    """Convert type hints to a more readable string."""
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if hasattr(type_hint, '__name__'):
        return type_hint.__name__.replace('NoneType', 'None')
    elif origin is typing.Union:
        if len(args) == 2 and type(None) in args:
            return f'Optional[{type_to_str(args[0])}]'
        else:
            return ' or '.join(type_to_str(arg) for arg in args)
    elif origin is typing.Callable:
        if args[0] is Ellipsis:
            args_str = '...'
        else:
            args_str = ', '.join(type_to_str(arg) for arg in args[:-1])
        return f'Callable[[{args_str}], {type_to_str(args[-1])}]'
    elif origin:
        inner_types = ', '.join(type_to_str(arg) for arg in args)
        origin_name = origin.__name__ if hasattr(origin, '__name__') else str(origin)
        return f'{origin_name}[{inner_types}]'
    else:
        return str(type_hint).replace('typing.', '')


def extract_comments_above_fields(dataclass_obj, prefix: str = '', level: int = 0, **kwargs):
    source_lines = inspect.getsource(dataclass_obj).split('\n')
    fields_info = {
        field.name: {
            'type': field.type,
            'default': field.default if field.default != MISSING else None,
            'default_factory': field.default_factory if field.default_factory != MISSING else None,
        }
        for field in fields(dataclass_obj)
    }
    comments, comment_cache = {}, []

    for line in source_lines:
        # skip unfinished multiline comments
        if line.count("'") == 3 or line.count('"') == 3:
            continue
        line_comment = extract_comments(line)
        if line_comment:
            comment_cache.append(line_comment[0])
        if ':' not in line:
            continue

        field_name = line.split(':')[0].strip()
        if field_name not in fields_info:
            continue

        field_info = fields_info[field_name]
        field_name = prefix + field_name
        field_type = type_to_str(field_info['type'])
        default = field_info['default']
        default_factory = field_info['default_factory']
        if default == '???':
            default_str = ' = MISSING'
        else:
            default_str = f' = {default}'
        if default_factory:
            try:
                default_factory = default_factory()
                default_str = f' = {default_factory}'
            except:
                pass
            if is_dataclass(default_factory):
                default_str = f' = {field_type}()'

        indent = '  ' * level
        comment = f"\n{indent}".join(comment_cache)
        comment = "- " + comment if comment else ""
        comment = comment.replace('\n', f'\n{indent}  ')
        field_detail = f"{indent}\033[92m{field_name}: {field_type}{default_str}\033[0m {comment}"
        comments[field_name] = field_detail
        comment_cache = []

        # Recursively extract nested dataclasses
        if is_dataclass(field_info['type']):
            nested_comments = extract_comments_above_fields(
                field_info['type'], prefix=field_name + '.', level=level + 1
            )
            for k, v in nested_comments.items():
                comments[f"{field_name}.{k}"] = v

    return comments


def get_fields_docstring(dataclass_obj, **kwargs):
    commented_fields = extract_comments_above_fields(dataclass_obj, **kwargs)
    docstring = [content for content in commented_fields.values()]
    return '\n'.join(docstring)


def get_help_message(dataclass_obj, help_message="", **kwargs):
    heading = """
This script uses Hydra (https://hydra.cc/) for dynamic configuration management.
You can apply Hydra's command-line syntax for overriding configuration values directly.
Below are the available configuration options and their default values:
    """.strip()

    docstring = get_fields_docstring(dataclass_obj)
    # to handle {} in docstring. Might need to add some other edge-case handling
    # here, so that formatting does not complain
    docstring = docstring.replace('{}', '{{}}')
    docstring = docstring.format(**kwargs)

    full_help = f"{heading}\n{'-' * 75}\n{docstring}"
    if help_message:
        full_help = f"{help_message}\n\n{full_help}"

    return full_help


def python_doc_to_cmd_help(doc_class, docs_prefix="", arg_prefix=""):
    """Converts python doc to cmd help format.

    Will color the args and change the format to match what we use in cmd help.
    """
    all_args = docs_prefix
    all_args += doc_class.__doc__.split("Args:")[1].rstrip()
    # \033[92m ... \033[0m - green in terminal
    colored_args = ""
    for line in all_args.split("\n"):
        if "        " in line and " - " in line:
            # add colors
            line = line.replace("        ", "        \033[92m").replace(" - ", "\033[0m - ")
            # fixing arg format
            line = line.replace('        \033[92m', f'        \033[92m{arg_prefix}')
        # fixing indent
        line = line.replace("        ", "    ").replace("    ", "  ")
        colored_args += line + '\n'
    return colored_args[:-1]
