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
from typing import Dict

from ansi2html import Ansi2HTMLConverter
from dash import dcc, html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def design_text_output(text: str, style={}):
    conv = Ansi2HTMLConverter()
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    if bool(ansi_escape.search(text)):
        text = conv.convert(text, full=False)
        return html.Div(
            iframe_template(
                '<link rel="stylesheet" type="text/css" href="assets/ansi_styles.css">',
                text,
            ),
            style=style,
        )

    def preprocess_latex(text):
        # Replace \[ and \] with $$ $$
        text = re.sub(r'\\\[', '\n$$', text)
        text = re.sub(r'\\\]', '$$\n', text)

        reg_exp = r'(?:\$\$(?:.|\n)*?\$\$|\$(?!\s)(?:[^$\n]|\\\$)*?(?<!\s)\$)'

        # Extract blocks inside $$ $$ and $ $
        math_blocks = re.findall(reg_exp, text, flags=re.DOTALL)

        # Replace the math blocks with placeholders
        for i, block in enumerate(math_blocks):
            text = text.replace(block, f'__MATH_BLOCK_{i}__')

        # Add $$ $$ around \begin{} and \end{} blocks
        text = re.sub(r'\\begin\{(.*?)\}', r'\n$$\\begin{\1}', text)
        text = re.sub(r'\\end\{(.*?)\}', r'\\end{\1}$$\n', text)

        # Replace the placeholders with the original math blocks
        for i, block in enumerate(math_blocks):
            text = text.replace(f'__MATH_BLOCK_{i}__', block)

        # Extract all blocks (including our custom blocks) inside $$ $$ and $ $
        math_blocks = re.findall(reg_exp, text, flags=re.DOTALL)

        # Replace the math blocks with placeholders
        for i, block in enumerate(math_blocks):
            text = text.replace(block, f'__MATH_BLOCK_{i}__')

        # Escape plain text outside of math blocks
        text = re.sub(r'([*_{}[\]()#+\-\.!])', r'\\\1', text)

        # Replace the placeholders with the original math blocks
        for i, block in enumerate(math_blocks):
            text = text.replace(f'\_\_MATH\_BLOCK\_{i}\_\_', block)

        return text.replace('\n', '\n\n')

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
