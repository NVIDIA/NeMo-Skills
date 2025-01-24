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
from collections import Counter

eval_funcs = {
    "add": lambda **k: k.get('p') + k.get('q'),
    "sub": lambda **k: k.get('p') - k.get('q'),
    "mul": lambda **k: k.get('p') * k.get('q'),
    "div": lambda **k: k.get('p') / k.get('q'),
    "percent": lambda **k: k.get('p') * k.get('q') / 100,
    "pow": lambda **k: k.get('p') ** k.get('q'),
    "sqrt": lambda **k: k.get('p') ** 0.5,
}


operations_map = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
}

inv_operations_map = {v: k for k, v in operations_map.items()}


def get_eval_func(op):
    func = eval_funcs[inv_operations_map[op]]
    return func


def get_op_counts(counter):
    return sum(counter.get(op, 0) for op in "+-/*")


def extract_expressions(text: str):
    start = 0
    cur_expr = []
    for idx, c in enumerate(text):
        prev_len = len(cur_expr)
        if c.isspace():
            if cur_expr:
                cur_expr.append(c)
        elif c == '.':
            if cur_expr and cur_expr[-1].isdigit():
                cur_expr.append(c)
            elif cur_expr:
                result = ''.join(cur_expr)
                yield result.rstrip(), start
        elif c.isdigit():
            cur_expr.append(c)
        elif c == '=' and not cur_expr:
            continue
        elif c in '+-/*=()':
            cur_expr.append(c)
        else:
            result = ''.join(cur_expr)
            counter = Counter(result)
            if get_op_counts(counter) >= 2:
                yield result.rstrip(), start
            cur_expr = []
        if prev_len == 0 and len(cur_expr) > 0:
            start = idx

    result = ''.join(cur_expr)
    counter = Counter(result)
    if get_op_counts(counter) >= 2:
        yield result.rstrip(), start


def tokenize(expression):
    token_pattern = r'(?<!\d)-\d+(\.\d+)?|\d+(\.\d+)?|\*\*|[+*/()-]'
    tokens = []
    for match in re.finditer(token_pattern, expression):
        token = match.group(0)
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))
    return tokens


def infix_to_postfix(tokens):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '**': 3}
    output = []
    stack = []

    for token, start, end in tokens:
        if re.fullmatch("(?<!\d)-\d+(\.\d+)?|\d+(\.\d+)?", token):
            output.append((token, start, end))
        elif token == '(':
            stack.append((token, start, end))
        elif token == ')':
            while stack and stack[-1][0] != '(':
                output.append(stack.pop())
            stack.pop()  # pop the '('
        else:
            while stack and stack[-1][0] != '(' and precedence.get(stack[-1][0], 0) >= precedence.get(token, 0):
                output.append(stack.pop())
            stack.append((token, start, end))

    while stack:
        output.append(stack.pop())

    return output


def evaluate_postfix_once(postfix):
    stack = []

    for token, start, end in postfix:
        if re.fullmatch("(?<!\d)-\d+(\.\d+)?|\d+(\.\d+)?", token):
            stack.append((float(token), start, end))
        else:
            b, b_start, b_end = stack.pop()
            a, a_start, a_end = stack.pop()
            eval_func = get_eval_func(token)
            result = eval_func(p=a, q=b)

            try:
                if int(result) == result:
                    result = int(result)
            except:
                result = float(result)

            return result, a_start, b_end


def solve_expression(expression):
    tokens = tokenize(expression)
    steps = [expression]

    while len(tokens) > 1:
        postfix = infix_to_postfix(tokens)
        res, start, end = evaluate_postfix_once(postfix)

        left = expression[:start]
        right = expression[end:]
        if left and left[-1] == '(' and right and right[0] == ')':
            left = left[:-1]
            right = right[1:]

        new_expression = left + str(res) + right
        steps.append(new_expression)
        expression = new_expression
        tokens = tokenize(expression)

    return steps


def merge_solution_steps(solution_steps):
    solution = []
    for step in solution_steps[:-1]:
        solution.append(re.sub(r"(?<!\d)(-\d+(\.\d+)?)", r"(\1)", step))
    solution.append(solution_steps[-1].strip())
    solution = " = ".join(solution)
    solution = re.sub(r"\s+", " ", solution)
    return solution
