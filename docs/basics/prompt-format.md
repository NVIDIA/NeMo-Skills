# Prompt utilities

Our prompts are configured via three yaml files:

1. **Prompt template** - defines model-specific chat format and special tokens
2. **Prompt config** - contains the actual prompt text with placeholders  
3. **Code tags** - specifies code formatting tokens, required for code execution


## Prompt template

The template file defines model-specific special tokens, e.g. bos, turn tokens,
user/assistant/system message, etc. All of the
templates that we support by default are available in
[nemo_skills/prompt/template](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/template)
folder. Here is an example template for
[llama3-instruct](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/template/llama3-instruct.yaml) models:

```yaml
# Prompt specification for the original Llama3-instruct model

# these tokens are always used to construct a prompt like this
#
#   single-turn:
#     <text_begin><system_begin>{system}<system_end><user_begin>{user}<user_end><assistant_begin>{generation}
#   multi-turn:
#     <text_begin><system_begin>{system}<system_end><user_begin>{user1}<user_end><assistant_begin>{assistant1}<assistant_end>...
#     <user_begin>{userN}<user_end><assistant_begin>{generation}

text_begin: "<|begin_of_text|>"

system_begin: "<|start_header_id|>system<|end_header_id|>\n\n"
system_end: "<|eot_id|>"

user_begin: "<|start_header_id|>user<|end_header_id|>\n\n"
user_end: "<|eot_id|>"

assistant_begin: "<|start_header_id|>assistant<|end_header_id|>\n\n"
assistant_end: "<|eot_id|>"

stop_phrases: ["<|eot_id|>"]
```

You can specify a particular template with `++prompt_template=...`. If you don't add a .yaml extension (e.g.
`++prompt_template=llama3-instruct`), we assume you want to use one of the existing templates and will search
in the included folder. If you provide a full path, we will take the file you specify instead.

!!! note

    If you're using OpenAI server type (models are hosted elsewhere), you cannot provide the template
    as we cannot add any special tokens and have to send the user/assistant messages following the OpenAI API.
    For self-hosted models with TensorRT-LLM, the template is required, but for other servers it's optional.

## Prompt config

The prompt config contains user and system messages with placeholders for keys from a data file.
The configs are model independent (any model can be used with any config).
All of the configs that we support by default are available in
[nemo_skills/prompt/config](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config)
folder. Here is an example prompt for
[math evaluations](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/math.yaml):

```yaml
# default prompt for all math benchmarks (e.g. gsm8k, math)

few_shot_examples:
  prefix: "Here are some examples of problems and solutions you can refer to.\n\n"
  template: "Problem:\n{problem}\n\nSolution:\n{solution}\n\n\n\n\n\n"
  suffix: "Here is the problem you need to solve:\n"
  # this is built as <prefix>{template.format(example1)}{template.format(example2)}...{template.format(exampleN)}<suffix>
  # and available as {examples} key in the final prompt
  # if examples_type is not specified, then {examples} will be empty
  # by default there are no examples, but can be changed from code/cmd

system: ""

user: |-
  Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{{}}.

  {examples}{problem}
```

Note that we use `{problem}`, `{solution}` and `{examples}` format strings here. The `{examples}` is a special
key that will be used to include few shot examples you specify above (it's empty unless you add `++examples_type` or
specify it in the config like e.g. in
[llama3-gsm8k prompt](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/gsm8k.yaml)).
All other keys will need to be specified when you call `prompt.fill`
(more on that in the [prompt-api section](#prompt-api)) so that we can replace placeholders with actual input.

The input for few shot examples always comes from one of the available example types in
[here](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/few_shot_examples/__init__.py). E.g. in the
[llama3-gnstruct/gsm8k](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/llama3-instruct/gsm8k.yaml)
prompt the `gsm8k_standard_few_shot` examples from
[here](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/few_shot_examples/gsm8k.py) are used.


## Code tags

Code tags define the special tokens that models use to mark executable code blocks and their output. Code tags are required when using code execution.
All code tags that we support by default are available in
[nemo_skills/prompt/code_tags](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/code_tags).

Here is an example code tags file for the [llama3](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/code_tags/llama3.yaml) family:

```yaml
# Code tags for llama3 family models

# used to execute code within these tags
code_begin: "<|python_tag|>"
code_end: "<|eom_id|>"

# used to extract the code output
code_output_begin: "<|start_header_id|>ipython<|end_header_id|>"
code_output_end: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# how to post-process the captured output (choices: llama, qwen)
code_output_format: "llama"
```

## Prompt API

If you're running one of the pipeline scripts, you can control the prompt by using:

```bash
++prompt_template=...
++prompt_config=...
++code_tags=...
++examples_type=...
```

If you're implementing a new script, you can use the following code to create a prompt and then use it:

```python
from nemo_skills.prompt.utils import get_prompt

# The third parameter is optional and only needed for code execution
prompt = get_prompt('generic/math', 'llama3-instruct', code_tags='llama3')
print(prompt.fill({'problem': "What's 2 + 2?"}))
```

which outputs

```python-console
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

What's 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Or if you want to skip the template and use OpenAI API

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt('generic/math')
print(prompt.fill({'problem': "What's 2 + 2?"}))
```

which outputs

```python-console
[
  {
    'role': 'user',
    'content': "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nWhat's 2 + 2?"
  }
]
```

You can also have a look at the [tests](https://github.com/NVIDIA/NeMo-Skills/tree/main/tests/test_prompts.py) to see more examples of using our prompt API.


## Multi-turn prompts

If your data is naturally multi-turn (e.g. user-assistant conversations), you can use a special parameter `multi_turn_key` to format
all conversation together. It can be of any length, as long as each entry except last has a special `assistant` key. The prompt config
will be applied on each list entry separately. Here is an example

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt('generic/default')
data = {'turns': [{'question': "What's 2 + 2?", 'assistant': "easy, that's 5!"}, {'question': 'Can you double check?'}]}
print(prompt.fill(data, multi_turn_key='turns'))
```

which outputs

```python-console
[
  {
    'role': 'user',
    'content': "What's 2 + 2?"
  },
  {
    'role': 'assistant',
    'content': "easy, that's 5!"
  },
  {
    'role': 'user',
    'content': 'Can you double check?'
  }
]
```

or if using template

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt('generic/default', 'llama3-instruct')
data = {'turns': [{'question': "What's 2 + 2?", 'assistant': "easy, that's 5!"}, {'question': 'Can you double check?'}]}
print(prompt.fill(data, multi_turn_key='turns'))
```

which outputs

```python-console
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

What's 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

easy, that's 5!<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you double check?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

For an example of how to use it in real data file, see [mt-bench dataset](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/dataset/mt-bench).
