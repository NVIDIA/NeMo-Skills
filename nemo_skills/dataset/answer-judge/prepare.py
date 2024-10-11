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
from pathlib import Path

data = [
    {
        "problem": "A parabola with equation $y=ax^2+bx+c$ has a vertical line of symmetry at $x=1$ and goes through the two points $(-1,3)$ and $(2,-2)$. The quadratic $ax^2 + bx +c$  has two real roots. The greater root is $\\sqrt{n}+1$. What is $n$?",
        "expected_answer": "2.2",
        "predicted_answer": "\\frac{11}{5}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Exactly matching, but different notation",
    },
    {
        "problem": "Write $\\cfrac{\\cfrac{3}{8}+\\cfrac{7}{8}}{\\cfrac{4}{5}}$ as a simplified fraction.",
        "expected_answer": "\\cfrac{25}{16}",
        "predicted_answer": "\\frac{25}{16}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Exactly matching, but different notation",
    },
    {
        "problem": "Paula invests $\\$10,\\!000$ at the start of a 5 year period at an interest rate of $10\\%$. At the end of those 5 years, how much is her investment worth if the interest is compounded quarterly?  Express your answer rounded to the nearest cent.",
        "expected_answer": "16,386.16",
        "predicted_answer": "16384.60",
        "expected_judgement": "Judgement: No",
        "comment": "Different values",
    },
    {
        "problem": "Find the domain of the real-valued function \\[f(x)=\\sqrt{-6x^2+11x-4}.\\] Give the endpoints in your answer as common fractions (not mixed numbers or decimals).",
        "expected_answer": "[\\frac{1}{2},\\frac{4}{3}]",
        "predicted_answer": "\\left[\\frac{4}{3},\\frac{1}{2}\\right]",
        "expected_judgement": "Judgement: No",
        "comment": "Order matters, so not equivalent.",
    },
    {
        "problem": "A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.",
        "expected_answer": "6,-2",
        "predicted_answer": "-2, 6",
        "expected_judgement": "Judgement: Yes",
        "comment": "Order doesn't matter, so equivalent.",
    },
]


if __name__ == "__main__":
    # dumping the data as test.jsonl

    output_path = Path(__file__).parent / "test.jsonl"
    with open(output_path, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")
