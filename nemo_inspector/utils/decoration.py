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
from difflib import SequenceMatcher
from html import escape
from io import StringIO
from typing import Callable, Dict, List, Optional, Tuple, Union

from ansi2html import Ansi2HTMLConverter
from dash import dcc, html
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from settings.constants import ANSI, COMPARE, LATEX, MARKDOWN


def color_text_diff(text1: str, text2: str) -> str:
    if text1 == text2:
        return [(text1, {})]
    matcher = SequenceMatcher(None, text1, text2)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.append((text2[j1:j2], {}))
        elif tag == 'replace':
            result.append((text1[i1:i2], {"background-color": "#c8e6c9"}))
            result.append((text2[j1:j2], {"background-color": "#ffcdd2", "text-decoration": "line-through"}))
        elif tag == 'insert':
            result.append((text2[j1:j2], {"background-color": "#ffcdd2", "text-decoration": "line-through"}))
        elif tag == 'delete':
            result.append((text1[i1:i2], {"background-color": "#c8e6c9"}))
    return result


def get_starts_with_tag_function(tag: str, default_index_move: int) -> Callable[[str, int], Tuple[bool, int]]:
    def starts_with_tag_func_templ(text: str, index: int):
        is_starts_with_tag = text.startswith(tag, index)
        if not is_starts_with_tag:
            returning_index = index + default_index_move
        elif '{' not in tag:
            returning_index = index + len(tag)
        else:
            returning_index = text.find('}', index) % (len(text) + 1)

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
        text[index] == '$' and not text[index + direction].isspace(),
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
    special_chars = r'*_{}[]()#+-.!`'
    for character in special_chars:
        text = text.replace(character, '\\' + character)
    return text


def preprocess_latex(text: str, escape: bool = True) -> str:
    text = '\n' + text.replace('\\[', '\n$$\n').replace('\\]', '\n$$\n').replace('\\(', ' $').replace('\\)', '$ ')

    right_side_operations = ['-', '=', '+', '*', '/']
    left_side_operations = ['=', '+', '*', '/']
    for op in right_side_operations:
        text = text.replace(op + '$', op + ' $')

    for op in left_side_operations:
        text = text.replace('$' + op, '$ ' + op)

    text += '\n'
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
                texts.append(
                    proccess_plain_text(text[start_plain_text_index:index])
                    if escape
                    else text[start_plain_text_index:index]
                )
                start_plain_text_index = -1

            start_index, new_index = proccess_tag(
                text,
                index,
                detect_start_token,
                detect_end_token,
                end_sign,
                use_last_block_only,
            )
            texts.append(proccess_plain_text(text[index:start_index]) if escape else text[index:start_index])
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
        texts.append(proccess_plain_text(text[start_plain_text_index:]) if escape else text[start_plain_text_index:])
    return ''.join(texts).replace('\n', '\n\n').strip()


def design_text_output(texts: List[Union[str, str]], style={}, text_modes: List[str] = [LATEX, ANSI]) -> html.Div:
    conv = Ansi2HTMLConverter()
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    full_text = ''.join(map(lambda x: x[0], texts))
    if ANSI in text_modes:
        if bool(ansi_escape.search(full_text)) or 'ipython-input' in full_text or 'Traceback' in full_text:
            if bool(ansi_escape.search(full_text)):
                full_text = conv.convert(full_text, full=False)
            else:
                full_text = conv.convert(full_text.replace('[', '\u001b['), full=False)
            return html.Div(
                iframe_template(
                    '<link rel="stylesheet" type="text/css" href="/assets/ansi_styles.css" href="/assets/compare_styles.css">',
                    f'<pre>{full_text}</pre>',
                ),
                style=style,
            )
    return html.Div(
        (
            dcc.Markdown(
                preprocess_latex(full_text, escape=MARKDOWN not in text_modes),
                mathjax=True,
                dangerously_allow_html=True,
            )
            if LATEX in text_modes and COMPARE not in text_modes
            else (
                dcc.Markdown(full_text)
                if MARKDOWN in text_modes and COMPARE not in text_modes
                else [html.Span(text, style={**inner_style, "whiteSpace": "pre-wrap"}) for text, inner_style in texts]
            )
        ),
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


def highlight_code(codes: List[Tuple[str, Dict[str, str]]], **kwargs) -> html.Iframe:

    full_code = ''.join([code for code, style in codes])

    # Track positions and styles
    positions = []
    current_pos = 0
    for code, style in codes:
        start_pos = current_pos
        end_pos = current_pos + len(code)
        if style:
            positions.append((start_pos, end_pos, style))
        current_pos = end_pos

    # Custom formatter to apply styles at correct positions
    class CustomHtmlFormatter(HtmlFormatter):
        def __init__(self, positions, **options):
            super().__init__(**options)
            self.positions = positions
            self.current_pos = 0

        def format(self, tokensource, outfile):
            style_starts = {start: style for start, _, style in self.positions}
            style_ends = {end: style for _, end, style in self.positions}
            active_styles = []

            for ttype, value in tokensource:
                token_length = len(value)
                token_start = self.current_pos

                # Apply styles character by character
                result = ''
                for i, char in enumerate(value):
                    char_pos = token_start + i

                    # Check if a style starts or ends here
                    if char_pos in style_starts:
                        style = style_starts[char_pos]
                        active_styles.append(style)
                    if char_pos in style_ends:
                        style = style_ends[char_pos]
                        if style in active_styles:
                            active_styles.remove(style)

                    # Get CSS class for syntax highlighting
                    css_class = self._get_css_class(ttype)
                    char_html = escape(char)
                    if css_class:
                        char_html = f'<span class="{css_class}">{char_html}</span>'

                    # Apply active styles
                    if active_styles:
                        combined_style = {}
                        for style_dict in active_styles:
                            combined_style.update(style_dict)
                        style_str = '; '.join(f'{k}: {v}' for k, v in combined_style.items())
                        char_html = f'<span style="{style_str}">{char_html}</span>'

                    result += char_html

                outfile.write(result)
                self.current_pos += token_length

    # Use the custom formatter to highlight the code
    lexer = PythonLexer()
    formatter = CustomHtmlFormatter(positions, nowrap=True)
    style_defs = formatter.get_style_defs('.highlight')
    style_defs += """
.highlight {
    font-family: monospace;
}
"""

    output = StringIO()
    formatter.format(lexer.get_tokens(full_code), output)
    highlighted_code = output.getvalue()

    # Build the iframe content
    iframe_id = get_random_id()
    content = f"""
<div class="highlight" style="white-space: pre-wrap; background-color: #ebecf0d8;">{highlighted_code}</div>
<script>{update_height_js(iframe_id)}</script>
"""

    return html.Div(
        iframe_template(
            header=f"<style>{style_defs}</style>",
            content=content,
            iframe_id=iframe_id,
            style={"border": "black 1px solid", "background-color": "#ebecf0d8"},
        )
    )
