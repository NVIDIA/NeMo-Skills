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

import json
import os
import random
import re

import gradio as gr
from latex2mathml.converter import convert
from latex2mathml.exceptions import NoAvailableTokensError


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def render_latex(text):
    def replace_matrix(match):
        matrix_content = match.group(1)
        rows = matrix_content.split('\\\\')
        mml_rows = ''.join(f'<mtr><mtd>{convert_and_clean(row.strip())}</mtd></mtr>' for row in rows)
        return f'<mrow><mo>(</mo><mtable rowspacing="4pt" columnspacing="1em">{mml_rows}</mtable><mo>)</mo></mrow>'

    def replace_align(match):
        align_content = match.group(1)
        rows = align_content.split('\\\\')
        mml_rows = []
        for row in rows:
            if '&' in row:
                left, right = row.split('&')
                mml_row = f'<mtr><mtd columnalign="right">{convert_and_clean(left.strip())}</mtd><mtd columnalign="left">{convert_and_clean(right.strip())}</mtd></mtr>'
            else:
                mml_row = f'<mtr><mtd columnalign="center">{convert_and_clean(row.strip())}</mtd></mtr>'
            mml_rows.append(mml_row)
        return f'<mtable columnspacing="1em" rowspacing="3pt" displaystyle="true">{"".join(mml_rows)}</mtable>'

    def convert_and_clean(latex):
        try:
            # Pre-process nested matrices
            latex = re.sub(r'\\begin{pmatrix}(.*?)\\end{pmatrix}', replace_matrix, latex, flags=re.DOTALL)

            # Handle \displaystyle
            latex = latex.replace('\\displaystyle', '')

            # Handle nested exponents
            latex = re.sub(r'\^{([^{}]+)}', r'^{\1}', latex)

            # Convert LaTeX to MathML
            mathml = convert(latex)
            mathml = re.sub(r'<math.*?>(.*)</math>', r'\1', mathml)
            return mathml
        except NoAvailableTokensError:
            return latex

    # Handle align* environment
    text = re.sub(
        r'\\begin{align\*}(.*?)\\end{align\*}',
        lambda m: f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{replace_align(m)}</math>',
        text,
        flags=re.DOTALL,
    )

    # Handle display math, excluding intervals
    text = re.sub(
        r'\[(?![-\d, ]+\])(.*?)\]',
        lambda m: f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{convert_and_clean(m.group(1))}</math>',
        text,
        flags=re.DOTALL,
    )

    # Handle inline math
    text = re.sub(
        r'\$(.*?)\$',
        lambda m: f'<math xmlns="http://www.w3.org/1998/Math/MathML">{convert_and_clean(m.group(1))}</math>',
        text,
    )

    return text


def display_entry(index, data_openmath2, data_math_train, current_test_set):
    entry_openmath2 = data_openmath2[index]
    entry_math_train = data_math_train[index]

    # Check if the current test set is GSM8K
    if current_test_set == "gsm8k":
        test_problem = entry_openmath2['problem']
        similar_openmath2 = entry_openmath2['similar_items']
        similar_math_train = entry_math_train['similar_items']
    else:
        test_problem = render_latex(entry_openmath2['problem'])
        similar_openmath2 = [render_latex(cand) for cand in entry_openmath2['similar_items']]
        similar_math_train = [render_latex(cand) for cand in entry_math_train['similar_items']]

    html = f"<h2>Test set problem:</h2><p>{test_problem}</p>"
    html += "<hr>"
    html += "<div style='display: flex;'>"
    html += "<div style='flex: 1; padding-right: 10px;'>"
    html += "<h2>Most similar OpenMathInstruct-2 problems:</h2><ol>"
    for cand in similar_openmath2:
        html += f"<li>{cand}</li>"
    html += "</ol></div>"
    html += "<div style='border-left: 1px solid #ccc;'></div>"
    html += "<div style='flex: 1; padding-left: 10px;'>"
    html += "<h2>Most similar MATH training set problems:</h2><ol>"
    for cand in similar_math_train:
        html += f"<li>{cand}</li>"
    html += "</ol></div>"
    html += "</div>"

    return html


def random_entry(data):
    return random.randint(0, len(data) - 1)


def load_test_sets(test_set):
    file_path_openmath2 = f'./similar-retrieved-openmath2/{test_set}'
    file_path_math_train = f'./similar-retrieved-math-train/{test_set}'

    data_openmath2 = load_jsonl(file_path_openmath2)
    data_math_train = load_jsonl(file_path_math_train)

    # Sort both datasets based on the 'problem' field (or use 'id' if available)
    data_openmath2.sort(key=lambda x: x['problem'])
    data_math_train.sort(key=lambda x: x['problem'])

    # Check if the sorted datasets have the same length and matching problems
    if len(data_openmath2) != len(data_math_train):
        raise ValueError(
            f"Datasets have different lengths: OpenMathInstruct-2 ({len(data_openmath2)}) vs MATH training set ({len(data_math_train)})"
        )

    for i, (entry_openmath2, entry_math_train) in enumerate(zip(data_openmath2, data_math_train)):
        if entry_openmath2['problem'] != entry_math_train['problem']:
            raise ValueError(
                f"Mismatch at index {i}: OpenMathInstruct-2 problem doesn't match MATH training set problem"
            )

    return data_openmath2, data_math_train


test_sets = [f for f in os.listdir('./similar-retrieved-openmath2') if f.endswith('.jsonl')]
test_set_names = [os.path.splitext(f)[0] for f in test_sets]

if "math.jsonl" in test_sets:
    test_sets.remove("math.jsonl")
    test_sets.insert(0, "math.jsonl")
    test_set_names = [os.path.splitext(f)[0] for f in test_sets]

with gr.Blocks() as demo:
    gr.Markdown("# OpenMathInstruct-2 test set contamination explorer")
    gr.Markdown(
        "See our full dataset at HuggingFace: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)"
    )

    warning_box = gr.Markdown(visible=False)

    with gr.Row():
        test_set_dropdown = gr.Dropdown(choices=test_set_names, label="Select Test Set", value=test_set_names[0])
        index_input = gr.Number(label="Problem Index", value=0, step=1)
        random_button = gr.Button("Random Problem")

    output = gr.HTML()

    data_openmath2 = gr.State(load_test_sets(test_sets[0])[0])
    data_math_train = gr.State(load_test_sets(test_sets[0])[1])
    current_test_set = gr.State(test_set_names[0])

    def update_test_set(test_set):
        new_data_openmath2, new_data_math_train = load_test_sets(f"{test_set}.jsonl")
        warning = ""
        warning_visible = False
        if test_set == "omni-math":
            warning = "⚠️ Since Omni-Math benchmarks was released after we finished training of our models, we didn't perform decontamination with it and some of the problems might match exactly!"
            warning_visible = True
        return (
            new_data_openmath2,
            new_data_math_train,
            0,
            display_entry(0, new_data_openmath2, new_data_math_train, test_set),
            warning,
            gr.update(visible=warning_visible),
            test_set,
        )

    def display_entry_wrapper(index, data_openmath2, data_math_train, current_test_set):
        return display_entry(index, data_openmath2, data_math_train, current_test_set)

    def random_entry_wrapper(data_openmath2):
        return random_entry(data_openmath2)

    test_set_dropdown.change(
        update_test_set,
        inputs=[test_set_dropdown],
        outputs=[data_openmath2, data_math_train, index_input, output, warning_box, warning_box, current_test_set],
    )
    index_input.change(
        display_entry_wrapper, inputs=[index_input, data_openmath2, data_math_train, current_test_set], outputs=output
    )
    random_button.click(random_entry_wrapper, inputs=[data_openmath2], outputs=index_input)

    demo.load(
        display_entry_wrapper, inputs=[index_input, data_openmath2, data_math_train, current_test_set], outputs=output
    )

demo.launch(debug=False, server_name='0.0.0.0', server_port=5005)
