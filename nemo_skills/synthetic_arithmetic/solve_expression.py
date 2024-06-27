import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parents[3]))

from nemo_skills.synthetic_arithmetic.utils import get_eval_func


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
