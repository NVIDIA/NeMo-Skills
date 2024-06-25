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

import random
import re
import string
from typing import Callable, Dict, Optional, Tuple

from ansi2html import Ansi2HTMLConverter
from dash import dcc, html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def get_starts_with_tag_function(tag: str, default_index_move: int) -> Callable[[str, int], Tuple[bool, int]]:
    def starts_with_tag_func_templ(text: str, index: int):
        is_starts_with_tag = text.startswith(tag, index)
        if not is_starts_with_tag:
            returning_index = index + default_index_move
        elif '{' not in tag:
            returning_index = index + len(tag)
        else:
            returning_index = text.find('}', index) % (len(text) + 1) + 1

        return is_starts_with_tag, returning_index

    return starts_with_tag_func_templ


def proccess_tag(
    text: str,
    start_index: int,
    detect_start_token: Callable[[str, int], Tuple[bool, int]],
    detect_end_token: Callable[[str, int], Tuple[bool, int]],
    end_sign: Optional[str],
    last_block_only: bool = False,
) -> int:
    count = 0
    index = start_index
    while index < len(text):
        if end_sign and text[index] == end_sign:
            return start_index, start_index + 1
        is_start_token, new_index = detect_start_token(text, index)
        count += is_start_token
        if last_block_only and is_start_token:
            start_index = index
            count = min(1, count)
        index = new_index
        is_end_token, index = detect_end_token(text, index)
        count -= is_end_token
        if count == 0:
            break
    return start_index, index + 1


def get_single_dollar_functions(direction: int, default_index_move: int) -> Callable[[str, int], Tuple[bool, int]]:
    return lambda text, index: (
        text[index] == '$' and text[index - direction].isspace() and not text[index + direction].isspace(),
        index + default_index_move,
    )


def get_detection_functions(text, index) -> tuple[
    Callable[[str, int], Tuple[bool, int]],
    Callable[[str, int], Tuple[bool, int]],
    Optional[str],
    bool,
    bool,
]:
    multiline_tags = [('\\begin{', '\\end{', True), ('$$', '$$', False)]
    for start_tag, end_tag, add_dollars in multiline_tags:
        if text.startswith(start_tag, index):
            return (
                get_starts_with_tag_function(start_tag, 1),
                get_starts_with_tag_function(end_tag, 0),
                None,
                add_dollars,
                False,
            )

    starts_with_dollar_func = get_single_dollar_functions(1, 1)
    ends_with_dollar_func = get_single_dollar_functions(-1, 0)
    if starts_with_dollar_func(text, index)[0]:
        return starts_with_dollar_func, ends_with_dollar_func, '\n', False, True

    return None, None, None, None, None


def proccess_plain_text(text: str) -> str:
    special_chars = r'*_{}[]()#+-.!'
    for character in special_chars:
        text = text.replace(character, '\\' + character)
    return text


def preprocess_latex(text: str) -> str:
    text = (
        '\n'
        + text.replace('\\[', '\n$$\n')
        .replace('\\]', '\n$$\n')
        .replace('=', ' = ')
        .replace('+', ' + ')
        .replace('-', ' - ')
        .replace('*', ' * ')
        .replace('/', ' / ')
        .replace('  ', ' ')
        + '\n'
    )
    index = 1
    texts = []
    start_plain_text_index = -1
    while index < len(text) - 1:
        (
            detect_start_token,
            detect_end_token,
            end_sign,
            add_dollars,
            use_last_block_only,
        ) = get_detection_functions(text, index)
        if detect_start_token is not None:
            if start_plain_text_index != -1:
                texts.append(proccess_plain_text(text[start_plain_text_index:index]))
                start_plain_text_index = -1

            start_index, new_index = proccess_tag(
                text,
                index,
                detect_start_token,
                detect_end_token,
                end_sign,
                use_last_block_only,
            )
            texts.append(proccess_plain_text(text[index:start_index]))
            if add_dollars:
                texts.append('\n$$\n')
                texts.append(text[start_index:new_index].strip())
                texts.append('\n$$\n')
            else:
                texts.append(text[start_index:new_index])
            index = new_index
        elif start_plain_text_index == -1:
            start_plain_text_index = index
            index += 1
        else:
            index += 1
    if start_plain_text_index != -1:
        texts.append(proccess_plain_text(text[start_plain_text_index:]))
    return ''.join(texts).replace('\n', '\n\n').strip()


def design_text_output(text: str, style={}) -> html.Div:
    conv = Ansi2HTMLConverter()
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    if bool(ansi_escape.search(text)):
        text = conv.convert(text, full=False)
        return html.Div(
            iframe_template(
                '<link rel="stylesheet" type="text/css" href="/assets/ansi_styles.css">',
                f'<pre>{text}</pre>',
            ),
            style=style,
        )
    elif 'ipython-input' in text or 'Traceback' in text:
        text = conv.convert(text.replace('[', '\u001b['), full=False)
        return html.Div(
            iframe_template(
                '<link rel="stylesheet" type="text/css" href="/assets/ansi_styles.css">',
                f'<pre>{text}</pre>',
            ),
            style=style,
        )

    return html.Div(
        dcc.Markdown(preprocess_latex(text), mathjax=True, dangerously_allow_html=True),
        style=style,
    )


def update_height_js(iframe_id: str) -> str:
    return f"""
        function updateHeight() {{
            var body = document.body,
                html = document.documentElement;
            
            var height = Math.max(body.scrollHeight, body.offsetHeight,
                                    html.clientHeight, html.scrollHeight, html.offsetHeight);
            
            parent.postMessage({{ frameHeight: height, frameId: '{iframe_id}' }}, '*');
        }}
        window.onload = updateHeight;
        window.onresize = updateHeight;
    """


def iframe_template(header: str, content: str, style: Dict = {}, iframe_id: str = None) -> html.Iframe:
    if not iframe_id:
        iframe_id = get_random_id()

    iframe_style = {
        "width": "100%",
        "border": "none",
        "overflow": "hidden",
    }

    iframe_style.update(style)

    return html.Iframe(
        id=iframe_id,
        srcDoc=f"""
        <!DOCTYPE html>
        <html>
        <head>
            {header}
        </head>
        <body>
            {content}
        </body>
        </html>
        """,
        style=iframe_style,
    )


def get_random_id() -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


def highlight_code(code: str) -> html.Iframe:
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())
    iframe_id = get_random_id()
    return iframe_template(
        header=f"<style>{HtmlFormatter().get_style_defs()}</style>",
        content=f"""<pre class='highlight'>{highlighted_code}</pre>
        <script>{update_height_js(iframe_id)}</script>""",
        iframe_id=iframe_id,
        style={"border": "black 1px solid", "background-color": "#ebecf0d8"},
    )
