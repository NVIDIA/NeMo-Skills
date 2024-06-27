import random
from collections import Counter

question_templates = {
    "add": [
        '{p} + {q}',
        '{p}+{q}',
        'Work out {p} + {q}.',
        'Add {p} and {q}.',
        'Sum of {p} and {q}.',
        'Add together {p} and {q}.',
        'What is {p} plus {q}?',
        'Calculate {p} + {q}.',
        'What is {p} + {q}?',
    ],
    "sub": [
        '{p} - {q}',
        '{p}-{q}',
        'Work out {p} - {q}.',
        'What is {p} minus {q}?',
        'Subtract {q} from {p}.',
        'Calculate {p} - {q}.',
        'What is {p} - {q}?',
    ],
    "mul": [
        '{p} * {q}',
        '{p}*{q}',
        '{p} \\cdot {q}',
        '{p} \\times {q}',
        'Calculate {p} * {q}.',
        'Work out {p}*{q}.',
        'Multiply {p} and {q}.',
        'Find product of {p} and {q}.',
        'What is the product of {p} and {q}?',
        '{p} times {q}',
        'What is {p} times {q}?',
    ],
    "div": [
        '{p} / {q}',
        '{p}/{q}',
        '{p} : {q}',
        '{p} \\div {q}',
        '{p} \\over {q}',
        '\\frac{{{p}}}{{{q}}}',
        'Divide {p} by {q}.',
        '{p} divided by {q}',
        'What is {p} divided by {q}?',
        'Calculate {p} divided by {q}.',
    ],
    "percent": [
        'What is {p}% of {q}?',
        'Calculate {p}% of {q}.',
        'Find {p}% of {q}.',
        '{p}% of {q}',
        'Work out {p}% of {q}.',
    ],
    "pow": [
        '{p} ** {q}',
        '{p}**{q}',
        '{p} ^ {q}',
        '{p}^{q}',
        'Calculate {p} to the power of {q}.',
        'What is {p} raised to the power of {q}?',
        '{p} to the power of {q}',
        'Find {p} ^ {q}.',
    ],
    "sqrt": [
        '\\sqrt{{{p}}}',
        '{p} ** 0.5',
        '{p}^0.5',
        '{p} ^ (1/2)',
        '{p}**(1/2)',
        'What is the square root of {p}?',
        'Calculate the square root of {p}.',
        'Find the square root of {p}.',
        'Square root of {p}',
    ],
    "multi": [
        '{eq}',
        'Compute {eq}',
        'What is {eq}?',
        'Calculate {eq}.',
        'Evaluate the expression: {eq}.',
        'Find the result of {eq}.',
        'Work out the value of {eq}.',
        'Determine the result of {eq}.',
    ],
}


solution_templates = {
    "add": "{p} + {q} = {ans}",
    "sub": "{p} - {q} = {ans}",
    "mul": "{p} * {q} = {ans}",
    "div": "{p} / {q} = {ans}",
    "percent": "{p} * {q} / 100 = {ans}",
    "pow": "{p} ** {q} = {ans}",
    "sqrt": "{p} ** 0.5 = {ans}",
}


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


def get_solution_template(op):
    template = solution_templates[op]
    return template


def get_template(op):
    template = random.choice(question_templates[op])
    return template


def get_str_repr(num):
    return str(num) if num >= 0 else f"({num})"


def is_valid_op(op, p, q):
    if op == "add":
        return p != 0 and q != 0
    elif op == "sub":
        return p != 0 and q != 0
    elif op == "mul":
        return abs(p * q) <= 1e4
    elif op == "div":
        return q != 0 and round(p / q, 1) == p / q
    elif op == "percent":
        return p >= 0 and round(q * p / 100, 1) == q * p / 100
    elif op == "pow":
        return 2 <= q <= 10 and -10000 <= p**q <= 10000
    elif op == "sqrt":
        return p > 1 and int(p**0.5) == p**0.5


def is_valid_multiturn_op(op, p, q):
    if op == "add":
        return True
    elif op == "sub":
        return True
    elif op == "mul":
        return abs(p * q) <= 1e4
    elif op == "div":
        return q != 0 and int(p / q) == p / q
    elif op == "pow":
        return 2 <= q <= 10 and -10000 <= p**q <= 10000
    else:
        return False


def augment_expression(expression):
    mul = random.choice(["*", "\\cdot", "\\times"])
    div = random.choice(["/", "\\div", ":"])
    space = random.choice([" "])
    expression = expression.replace("*", mul)
    expression = expression.replace("/", div)
    expression = expression.replace(" ", space)
    return expression


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
