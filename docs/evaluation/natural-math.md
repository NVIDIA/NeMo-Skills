# Math (natural language)

We support a variety of natural language math benchmarks. For all benchmarks in this group the task
is to find an answer to a math problem. This is typically a number or an expression that an LLM is instructed
to put inside a `\boxed{}` field.

By default all benchmarks in this group use
[generic/math](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) prompt config.

## How we compare answers

Most answers in these benchmarks can be compared using a
[symbolic checker](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/math_grader.py#L47)
but a few require using LLM-as-a-judge. By default those benchmarks will use GPT-4.1 and thus require OPENAI_API_KEY
to be defined. If you want to host a local judge model instead, you can change benchmark parameters like this

```bash
    --judge_model=Qwen/Qwen2.5-32B-Instruct
    --judge_server_type=sglang
    --judge_server_gpus=2
```

You can see the full list of supported judge parameters by running `ns eval --help | grep "judge"`.

!!! note
    The judge task is fairly simple, it only needs to compare expected and predicted answers in the context of the problem.
    It **does not** need to check the full solution for correctness. By default we use
    [judge/math](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/judge/math.yaml) prompt for the judge.

The following benchmarks require LLM-as-a-judge:

- [omni-math](#omni-math)
- [math-odyssey](#math-odyssey)
- [gaokao2023en](#gaokao2023en)

## How we extract answers

By default we will extract the answer from the last `\boxed{}` field in the generated solution. This is consistent
with our default [generic/math](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) prompt config.

We also support arbitrary regex based extraction. E.g., if you use a custom prompt that asks an LLM to put an answer after `Final answer:`
at the end of the solution, you can use these parameters to match the extraction logic to that prompt

```bash
    --extra_eval_args="++eval_config.extract_from_boxed=False ++eval_config.extract_regex='Final answer: (.+)$'"
```

!!! warning
    Most LLMs are trained to put an answer for math problems inside `\boxed{}` field. For many models even if you ask
    for a different answer format in the prompt, they might not follow this instruction. We thus generally do not
    recommend changing extraction logic for these benchmarks.

## Supported benchmarks

### aime25

- Benchmark is defined in [`nemo_skills/dataset/aime25/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/aime25/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).

### aime24

- Benchmark is defined in [`nemo_skills/dataset/aime24/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/aime24/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).

### hmmt_feb25

- Benchmark is defined in [`nemo_skills/dataset/hmmt_feb25/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/hmmt_feb25/__init__.py)
- Original benchmark source is [here](https://www.hmmt.org/www/archive/282).

### brumo25

- Benchmark is defined in [`nemo_skills/dataset/brumo25/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/brumo25/__init__.py)
- Original benchmark source is [here](https://www.brumo.org/archive).

### comp-math-24-25

- Benchmark is defined in [`nemo_skills/dataset/comp-math-24-25/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/comp-math-24-25/__init__.py)
- This benchmark is created by us! See [https://arxiv.org/abs/2504.16891](https://arxiv.org/abs/2504.16891) for more details.

### omni-math

- Benchmark is defined in [`nemo_skills/dataset/omni-math/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/omni-math/__init__.py)
- Original benchmark source is [here](https://omni-math.github.io/).

### math

- Benchmark is defined in [`nemo_skills/dataset/math/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/math/__init__.py)
- Original benchmark source is [here](https://github.com/hendrycks/math).

### math-500

- Benchmark is defined in [`nemo_skills/dataset/math-500/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/math-500/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

### gsm8k

- Benchmark is defined in [`nemo_skills/dataset/gsm8k/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/gsm8k/__init__.py)
- Original benchmark source is [here](https://github.com/openai/grade-school-math).

### amc23

- Benchmark is defined in [`nemo_skills/dataset/amc23/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/amc23/__init__.py)
- Original benchmark source is [here](https://artofproblemsolving.com/wiki/index.php/2023_AMC_12A).

### college_math

- Benchmark is defined in [`nemo_skills/dataset/college_math/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/college_math/__init__.py)
- Original benchmark source is [here](https://github.com/XylonFu/MathScale).

### gaokao2023en

- Benchmark is defined in [`nemo_skills/dataset/gaokao2023en/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/gaokao2023en/__init__.py)
- Original benchmark source is [here](https://github.com/OpenLMLab/GAOKAO-Bench).

### math-odyssey

- Benchmark is defined in [`nemo_skills/dataset/math-odyssey/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/math-odyssey/__init__.py)
- Original benchmark source is [here](https://github.com/protagolabs/odyssey-math).

### minerva_math

- Benchmark is defined in [`nemo_skills/dataset/minerva_math/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/minerva_math/__init__.py)
- Original benchmark source is [here](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation/data/minerva_math).

### olympiadbench

- Benchmark is defined in [`nemo_skills/dataset/olympiadbench/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/olympiadbench/__init__.py)
- Original benchmark source is [here](https://github.com/OpenBMB/OlympiadBench).

### algebra222

- Benchmark is defined in [`nemo_skills/dataset/algebra222/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/algebra222/__init__.py)
- Original benchmark source is [here](https://github.com/joyheyueya/declarative-math-word-problem).

### asdiv

- Benchmark is defined in [`nemo_skills/dataset/asdiv/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/asdiv/__init__.py)
- Original benchmark source is [here](https://github.com/chaochun/nlu-asdiv-dataset).

### gsm-plus

- Benchmark is defined in [`nemo_skills/dataset/gsm-plus/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/gsm-plus/__init__.py)
- Original benchmark source is [here](https://github.com/qtli/GSM-Plus).

### mawps

- Benchmark is defined in [`nemo_skills/dataset/mawps/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mawps/__init__.py)
- Original benchmark source is [here](https://github.com/sroy9/mawps).

### svamp

- Benchmark is defined in [`nemo_skills/dataset/svamp/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/svamp/__init__.py)
- Original benchmark source is [here](https://github.com/arkilpatel/SVAMP).

### beyondaime

- Benchmark is defined in [`nemo_skills/dataset/svamp/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/beyondaime/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME).