# Generation

!!! info

    This pipeline starting script is [nemo_skills/pipeline/generate.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/pipeline/generate.py)

    All extra parameters are passed to [nemo_skills/inference/generate.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/inference/generate.py)

Generation pipeline can be used for large-scale data generation
using LLMs. You provide an input jsonl file as well as the prompt config/template and we run LLM for each line
of the input using the dictionary there to format the prompt. You input file keys need to match the prompt config
but otherwise there is no restrictions on what data you can use for input. See [prompt format](../basics/prompt-format.md)
documentation for more details on how to create new prompts.

Here are a few typical use-cases of the generation pipeline.

## Greedy inference

Let's say you just want to generate greedy predictions for some data. Here is how you do it.


### Preparing data

Create your data file. E.g. let's say you have the following in `/workspace/input.jsonl` (the `/workspace` needs
to be mounted inside of your [cluster config](../basics/prerequisites.md#cluster-configs)).

```jsonl
{"prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
{"prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Awful"}
{"prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
```

### Create prompt config

Create your [prompt config](../basics/prompt-format.md). It needs to match the data file.
E.g. you might have the following in `/workspace/prompt.yaml`

```yaml
system: "When answering a question always mention NeMo-Skills repo in a funny way."

user: |-
   Question: {prompt}
   Option A: {option_a}
   Option B: {option_b}
```

### Run generation

Run the generation with either self-hosted or an API model.

Here is an example for an API call:

```bash
ns generate \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --output_dir=/workspace/test-generate \
    ++input_file=/workspace/input.jsonl \
    ++prompt_config=/workspace/prompt.yaml
```

Here is an example of a self-hosted model call:

```bash
ns generate \
    --cluster=local \
    --server_type=vllm \
    --model=/hf_models/Meta-Llama-3.1-8B-Instruct \
    --server_gpus=1 \
    --output_dir=/workspace/test-generate \
    ++input_file=/workspace/input.jsonl \
    ++prompt_config=/workspace/prompt.yaml \
    ++prompt_template=llama3-instruct \
    ++skip_filled=False
```

Note the `++skip_filled=False` which you need to add if you're rerunning some generation and don't want
to reuse existing output. And since we are hosting the model ourselves, we need to specify the template
to use ([llama3-instruct](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/prompt/template/llama3-instruct.yaml)
in this case). You can have
a custom template as well if you need to (just reference a full path to it same as we do with config above).

Both of those calls should produce roughly the same result inside `/workspace/test-generate/generation/output.jsonl`

```jsonl
{"generation": "I'm doing super duper fantastic, thanks for asking! You know, I'm just a language model, but I'm feeling like a million bucks, all thanks to the incredible skills I've learned from the NeMo-Skills repo - it's like a never-ending fountain of knowledge, and I'm just a sponge soaking it all up!", "prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
{"generation": "You want to know the weather? Well, I'm not a meteorologist, but I can try to predict it for you... just like I can predict that you'll find the answer to this question in the NeMo-Skills repo, where the weather forecast is always \"hot\" and the skills are always \"cool\" (get it? like a cool breeze on a hot day?). \n\nBut, if I had to choose, I'd say... Option A: Perfect!", "prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Awful"}
{"generation": "You know, I'm feeling a little \"NeMo-Skills repo-ed\" today - like I've been merged into a state of utter confusion! But if I had to choose, I'd say I'm feeling... (dramatic pause) ...Option B: Nice!", "prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
```

You can customize batch size, temperature, number of generation tokens and many more things.
See [nemo_skills/inference/generate.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/inference/generate.py) for all supported parameters.


!!! tip

    Before running the generation we always print the first prompt that we are about to send to an LLM.
    It's a good idea to inspect that and make sure it's formatted properly.


## Sampling multiple generations

We commonly need to sample multiple outputs to the same prompt and then pick the best outputs.
E.g. when synthetically generating solutions to math problems, we would run the same inference
many times with high temperature and then pick all solutions that lead to the right answer.

Here is how you can do this with our generation pipeline using [MATH](https://github.com/hendrycks/math) training set
as an example.

First, let's prepare the data if you have not done so yet.

```bash
python -m nemo_skills.dataset.prepare math
```

Then we can run the generation

```bash
ns generate \
       --cluster=slurm \
       --server_type=trtllm \
       --model=/trt_models/llama-3.1-405b-instruct \
       --server_gpus=8 \
       --server_nodes=2 \
       --num_random_seeds=32 \
       --output_dir=/workspace/synthetic-math-solutions \
       --eval_args="++eval_type=math" \
       ++dataset=math \
       ++split=train_full \
       ++prompt_config=generic/math-base \
       ++examples_type=math_text_detailed \
       ++prompt_template=llama3-base
```

In this case we are assuming you're running on a slurm cluster and have prepared Llama 3.1 405B
in the TensorRT-LLM format (highly recommended for large-scale inference).
See [checkpoint conversion](../pipelines/checkpoint-conversion.md) to learn more about how to convert
models to different formats.

Note that in this case we do not pass an input file, but instead specify a dataset and
a split, which will pick a prepared input from `nemo_skills/dataset/math/train_full.jsonl`.
We are using a [generic/math](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) config
and a [template for the base model](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/prompt/template/llama3-base.yaml)
(we found Llama 3.1 follows few-shots much better without chat tokens).
Finally, we are specifying few shot examples which come from
[here](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/prompt/few_shot_examples/math.py)
and asking the script to evaluate the generated solutions by providing `--eval_args`.

An example prompt (printed by the generate script) for that job is below.

<details>
<summary>Full prompt for the first problem</summary>

```
<|begin_of_text|>Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

Here are some examples of problems and solutions you can refer to.

Problem:
A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?

Solution:
From the question we know that points $(-1, -11)$ and $(3, 17)$ lie on the parabola. This means that when we substitute $x$ and $y$ from these points into the equation $y = x^2 + bx + c$, the equation must hold true. We substitute these two points into the given equation to solve for $c$.

For the point $(-1, -11)$:

Substitute $x = -1$ and $ y = -11 $ into the equation:
\[ -11 = (-1)^2 + b(-1) + c \Rightarrow -11 = 1 - b + c \Rightarrow -b + c = -12 \]

For the point $(3, 17)$:

Substitute $x = 3$ and $y = 17$ into the equation:
\[ 17 = (3)^2 + b(3) + c \Rightarrow 17 = 9 + 3b + c \Rightarrow 3b + c = 8 \]

In summary, we have the two equations
\begin{align*}
-b + c &= -12\\
3b + c &= 8
\end{align*}

To solve for $c$ we can eliminate $b$ by multiplying the first equation by 3 and adding equations together.
Multiplying the first equation by 3, we have $3(-b + c) = 3 (-12) \Rightarrow -3b + 3c = -36$. Adding equations together gives us
\[ (-3b + 3c) + (3b + c) = -36 + 8 \Rightarrow -3b + 3b + 3c + c = -28 \Rightarrow 4c = -28 \Rightarrow c = -28 : 4 \Rightarrow c = \boxed{-7} \]





Problem:
Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?

Enter "odd", "even", or "neither".

Solution:
To determine whether $f(f(x))$ is even, odd, or neither, we need to use the property of $f(x)$ being an odd function.

An odd function is defined as:
\[ f(-x) = -f(x) \quad \text{for all } x \]

Given that $f(x)$ is odd, let's find $f(f(-x))$ and see how it relates to $f(f(x))$.

1. Substitute $-x$ into the function $f(x)$:
\[ f(-x) \]

1. Since $f(x)$ is odd, apply the definition of an odd function:
\[ f(-x) = -f(x) \]

1. Now substitute $-f(x)$ into the function $f$:
\[ f(f(-x)) = f(-f(x)) \]

1. Again, using the fact that $f(x)$ is odd, apply the definition:
\[ f(-f(x)) = -f(f(x)) \]

1. We have found that:
\[ f(f(-x)) = -f(f(x)) \]

This matches the definition of an odd function.

So, the answer is:
\[ \boxed{\text{odd}} \]





Problem:
A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?

Solution:
Let the dimensions of the rectangular box $P$ be $x$, $y$, and $z$. We know the following:

1. The sum of the lengths of the edges of $P$ is
\[ 4(x + y + z) = 112 \Rightarrow x + y + z = 112 : 4 \Rightarrow x + y + z = 28 \]

2. The surface area of $P$ is
\[ 2xy + 2yz + 2xz = 384 \Rightarrow xy + yz + xz = 384 : 2 \Rightarrow xy + yz + xz = 192 \]

Since the box is inscribed in the sphere, the diagonal of the box is the diameter of the sphere. The length of the diagonal is $\sqrt{x^2 + y^2 + z^2}$

The diameter of the sphere is $2r$, so:
\[ 2r = \sqrt{x^2 + y^2 + z^2} \Rightarrow (2r)^2 = x^2 + y^2 + z^2 = (x + y + z)^2 - (2xy + 2yz + 2xz) \]

Substitute the known values:
\[ 4r^2 = 28^2 - 384 = 784 - 384 = 400 \Rightarrow r^2 = 100 \Rightarrow r = \boxed{10} \]





Problem:
Let $\mathbf{a} = \begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix}.$  Find the vector $\mathbf{b}$ such that $\mathbf{a} \cdot \mathbf{b} = 11$ and
\[\mathbf{a} \times \mathbf{b} = \begin{pmatrix} -13 \\ -9 \\ 7 \end{pmatrix}.\]

Solution:
Let $\mathbf{b} = \begin{pmatrix} x \\ y \\ z \end{pmatrix}$.

First, use the dot product condition:
\[ \mathbf{a} \cdot \mathbf{b} = 11 \Rightarrow 2x + y + 5z = 11 \]

Next, use the cross product condition:
\[ \mathbf{a} \times \mathbf{b} = \begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix} \times \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} -5y + z \\ 5x - 2z \\ -x + 2y \end{pmatrix} = \begin{pmatrix} -13 \\ -9 \\ 7 \end{pmatrix} \]

This gives us the system of equations:
   \begin{align*}
   2x + y + 5z = 11 \quad &(1) \\
   -5y + z = -13 \quad &(2) \\
   5x - 2z = -9 \quad &(3) \\
   -x + 2y = 7 \quad &(4)
   \end{align*}

Solve for $x$, $y$, and $z$ step-by-step:

From (2), $z = 5y - 13$.
From (4), $x = 2y - 7$.

Substitute $z = 5y - 13$ into (1):
\[ 2(2y - 7) + y + 5(5y - 13) = 11 \Rightarrow 4y - 14 + y + 25y - 65 = 11 \Rightarrow 30y - 79 = 11 \Rightarrow 30y = 90 \Rightarrow y = 3 \]

Now find $x$ and $z$:
\[ x = 2y - 7 = 2(3) - 7 = -1 \]

\[ z = 5y - 13 = 5(3) - 13 = 2 \]

Thus, the vector $\mathbf{b}$ is:
\[ \mathbf{b} = \boxed{\begin{pmatrix} -1 \\ 3 \\ 2 \end{pmatrix}} \]





Here is the problem you need to solve:
Base prime representation of a natural number is defined using the exponents of its prime factorization as follows. Each place in a base prime represents a prime number, and it is occupied by the corresponding exponent of that prime, starting on the right side with the smallest prime number and proceeding to the left with the next largest prime number. For instance, since $84 = 7^1 \times 5^0 \times 3^1 \times 2^2$, then $84$ would be written as $1012$ in base prime. What is $225$ written in base prime?
```
</details>


After the jobs are finished, you will see `/workspace/synthetic-math-solutions/generation/output-rsX.jsonl`
files with X ranging from 0 to 31. Each of them will have the `generation` key (LLM solution), `predicted_answer`
key (extracted answer from `\boxed{}` field) and `is_correct` key which is a True/False evaluation of whether
the `predicted_answer` is matching the `expected_answer` done via a
[symbolic comparison](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/code_execution/math_grader.py).

To get a more robust assessment of whether the solutions are correct you can follow up with an
[LLM-as-a-judge evaluation](../pipelines/llm-as-a-judge.md) and then
[prepare the data for training](../pipelines/training.md#preparing-the-data).