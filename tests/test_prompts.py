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

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Meta AI, a sophisticated and energetic AI Assistant. You excel at solving mathematical problems.

You will look at the examples provided by the user and try to follow the solution format as much as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>

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
2 + 2 = ?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
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
    prompt = Prompt(config=get_prompt_config('nemotron/zeroshot'))
    expected_prompt = """<extra_id_0>System

<extra_id_1>User
2 + 2 = ?
<extra_id_1>Assistant
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_nemotron_fewshot_prompt():
    config = get_prompt_config('nemotron/fewshot')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!"},
    ]
    config.few_shot_examples.num_few_shots = 2
    prompt = Prompt(config=config)

    expected_prompt = """<extra_id_0>System

<extra_id_1>User
Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

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



Don't forget that your final answer should be inside \\boxed{}!
<extra_id_1>Assistant
"""
    assert prompt.build_string({'question': '2 + 2 = ?'}) == expected_prompt


def test_nemotron_fewshot_prompt_reference():
    config = get_prompt_config('nemotron/fewshot')
    config.few_shot_examples.example_dicts = [
        {'question': '1 + 1 = ?', 'generation': "That's easy: 2!", 'reference_solution': "Think hard - it's 2"},
        {'question': '5 + 5 = ?', 'generation': "That's easy: 10!", 'reference_solution': "Isn't it 10?"},
    ]
    config.few_shot_examples.num_few_shots = 2
    config.context_type = 'reference_solution'
    prompt = Prompt(config=config)

    expected_prompt = """<extra_id_0>System

<extra_id_1>User
Here are some examples of questions and solutions followed by a new question that you need to solve.
Make sure to put the answer (and only answer) inside \\boxed{}.

Example question:
1 + 1 = ?

Reference solution (do not copy it):
Think hard - it's 2

Example solution:
That's easy: 2!





Example question:
5 + 5 = ?

Reference solution (do not copy it):
Isn't it 10?

Example solution:
That's easy: 10!





Question:
2 + 2 = ?

Reference solution (do not copy it):
What should I do??



Don't forget that your final answer should be inside \\boxed{}!
<extra_id_1>Assistant
"""
    assert (
        prompt.build_string({'question': '2 + 2 = ?', 'reference_solution': 'What should I do??'}) == expected_prompt
    )
