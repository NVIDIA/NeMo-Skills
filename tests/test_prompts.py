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


from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config


def test_question_generation_rephrasing_prompt():
    config = get_prompt_config('question_generation/rephrasing')
    config.few_shot_examples.example_dicts = [
        {
            'question': 'Are you sure you want to do that?',
            'rephrased_question': "Is this really what you want to do?",
        },
        {'question': 'How are you?', 'rephrased_question': "How is it going?"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """You are an AI assistant that excels at rephrasing questions. Follow the given examples.

Question:
Are you sure you want to do that?

Rephrase the above question:
Is this really what you want to do?





Question:
How are you?

Rephrase the above question:
How is it going?





Question:
What's the meaning of life?

Rephrase the above question:
"""
    assert prompt.build_string({'question': "What's the meaning of life?"}) == expected_prompt


def test_question_generation_augmentation_prompt():
    config = get_prompt_config('question_generation/augmentation')
    config.few_shot_examples.example_dicts = [
        {
            'question': 'Are you sure you want to do that?',
            'augmented_question': "Is this really what you want to do?",
        },
        {'question': 'How are you?', 'augmented_question': "How is it going?"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """You are an AI assistant that excels at creating similar questions. Follow the given examples.

Question:
Are you sure you want to do that?

Write another question similar to this one:
Is this really what you want to do?





Question:
How are you?

Write another question similar to this one:
How is it going?





Question:
What's the meaning of life?

Write another question similar to this one:
"""
    assert prompt.build_string({'question': "What's the meaning of life?"}) == expected_prompt


def test_llama3_instruct_prompt():
    config = get_prompt_config('llama3/instruct')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Question:
1 + 1 = ?

My solution:
That's easy: 2!





Question:
5 + 5 = ?

My solution:
That's easy: 10!





Question:
2 + 2 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_llama3_base_prompt():
    config = get_prompt_config('llama3/base')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """<|begin_of_text|>Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Question:
1 + 1 = ?

My solution:
That's easy: 2!





Question:
5 + 5 = ?

My solution:
That's easy: 10!





Question:
2 + 2 = ?

My solution:
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_openmathinstruct_base_prompt():
    config = get_prompt_config('openmathinstruct/base')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Question:
1 + 1 = ?

My solution:
That's easy: 2!





Question:
5 + 5 = ?

My solution:
That's easy: 10!





Question:
2 + 2 = ?

My solution:
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_openmathinstruct_sft_prompt():
    prompt = Prompt(config=get_prompt_config('openmathinstruct/sft'))
    expected_prompt = """System:
You're an expert Python programmer and mathematician. Help the user to solve this problem using code when necessary. Make sure to put the answer (and only answer) inside \\boxed{}.

User:
2 + 2 = ?

Assistant:
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_nemotron_zeroshot_prompt():
    prompt = Prompt(config=get_prompt_config('nemotron/instruct'))
    expected_prompt = """<extra_id_0>System

<extra_id_1>User
Help the user to solve the given problem.

2 + 2 = ?
<extra_id_1>Assistant
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_nemotron_fewshot_prompt():
    config = get_prompt_config('nemotron/fewshot_instruct')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """<extra_id_0>System

<extra_id_1>User
Here are some examples of questions and solutions followed by a new question that you need to solve.

Example question:
1 + 1 = ?

Example solution:
That's easy: 2!





Example question:
5 + 5 = ?

Example solution:
That's easy: 10!





Question:
2 + 2 = ?
<extra_id_1>Assistant
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_llama3_gsm8k_prompt():
    config = get_prompt_config('llama3/gsm8k')
    prompt = Prompt(config=config)

    expected_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Milly needs to return a book she decided was really boring. The book weighs 4 pounds, cost $32, and needs to be returned to a distribution center 20 miles away. If the shipping company charges $0.35 per pound plus $0.08 per mile, and Amazon will only refund 75% of the book's purchase price, how much money will Milly lose?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert (
        prompt.build_string(
            {
                'question': "Milly needs to return a book she decided was really boring. The book weighs 4 pounds, cost $32, and needs to be returned to a distribution center 20 miles away. If the shipping company charges $0.35 per pound plus $0.08 per mile, and Amazon will only refund 75% of the book's purchase price, how much money will Milly lose?"
            }
        )
        == expected_prompt
    )


def test_llama3_math_prompt():
    config = get_prompt_config('llama3/math')
    prompt = Prompt(config=config)

    expected_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: Find the sum of all complex values of $a,$ such that the polynomial $x^4 + (a^2 - 1) x^2 + a^3$ has exactly two distinct complex roots.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert (
        prompt.build_string(
            {
                'question': "Find the sum of all complex values of $a,$ such that the polynomial $x^4 + (a^2 - 1) x^2 + a^3$ has exactly two distinct complex roots."
            }
        )
        == expected_prompt
    )


def test_llama3_mmlu_prompt():
    config = get_prompt_config('llama3/mmlu')
    prompt = Prompt(config=config)

    expected_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Given the following question and four candidate answers (A, B, C and D), choose the best answer.

Question: A 26-year-old woman is brought to the emergency department because of an 8-hour history of severe back and abdominal pain and mild but persistent vaginal bleeding. Ultrasonography of the abdomen shows a 2-cm ectopic pregnancy in the ampulla. The ampulla has ruptured into the surrounding tissue. Fluid from this rupture will most likely be found in which of the following locations?
A. Lesser peritoneal cavity
B. Mesometrium
C. Pouch of Douglas
D. Uterine cavity

- For simple problems:
Directly provide the answer with minimal explanation.

- For complex problems:
Use this step-by-step format:
## Step 1: [Concise description]
[Brief explanation]
## Step 2: [Concise description]
[Brief explanation]

Regardless of the approach, always conclude with:
The best answer is [the_answer_letter].
where the [the_answer_letter] is one of A, B, C or D.

Let's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert (
        prompt.build_string(
            {
                'question': "A 26-year-old woman is brought to the emergency department because of an 8-hour history of severe back and abdominal pain and mild but persistent vaginal bleeding. Ultrasonography of the abdomen shows a 2-cm ectopic pregnancy in the ampulla. The ampulla has ruptured into the surrounding tissue. Fluid from this rupture will most likely be found in which of the following locations?",
                'A': "Lesser peritoneal cavity",
                'B': "Mesometrium",
                'C': "Pouch of Douglas",
                'D': "Uterine cavity",
            }
        )
        == expected_prompt
    )
