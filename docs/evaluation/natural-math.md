# Math (natural language)

We support a variety of natural language math benchmarks. For all benchmarks in this group, the task is to find an answer to a math problem. This is typically a number or an expression that an LLM during generation step is instructed
to put inside a `\boxed{}` field.

Most answers in these benchmarks can be compared using a
[symbolic checker](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/math_grader.py#L47) but a few require using LLM-as-a-judge to evaluate the equivalency of the epxressions. We will see below how we can use LLM-as-a-judge in the `eval` pipeline with NeMo-Skills.

We have seen in [Evaluation](index.md) how to use `ns eval` to perform a benchmark evaluation which uses symbolic checker. To add LLM-as-judge in to the evaluation, we just need to add extra judge related parameters that serve as parameters for LLM-as-judge model.

=== "ns interface"

    ```bash
    ns eval \
        --cluster=local \
        --input_file=/workspace/input.jsonl \
        --server_type=openai \
        --output_dir=/workspace/generation-local-trtllm \
        --model=meta/llama-3.1-8b-instruct \
        --server_gpus=1 \
        --server_address=https://integrate.api.nvidia.com/v1
        --benchmarks=aime25 \
        --judge_model=nvidia/llama-3.1-nemotron-ultra-253b-v1 \
        --judge_server_address=https://integrate.api.nvidia.com/v1 \
        --judge_server_type=openai
    ```

=== "python interface"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

     eval(
        ctx=wrap_arguments(ctx_args),
        cluster="local",
        output_dir="/workspace/evaluation-local-trtllm",
        input_file="/workspace/input.jsonl",
        server_type="openai",
        model="meta/llama-3.1-8b-instruct",
        server_gpus=1,
        server_address="https://integrate.api.nvidia.com/v1",
        benchmarks="aime25",
        judge_model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        judge_server_address="https://integrate.api.nvidia.com/v1",
        judge_server_type="openai"

    )
    ```

Alternatively, you can move judge related parameters completely from the evaluation command into the dataset directly. For example, consider [hle benchmark](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/hle/__init__.py). This lets you define judge parameters specific to each benchmark separately.

 In the benchmark definition `__init__.py` script, you can add judge related parameters, which will be used as default LLM-as-judge model parameters.

```bash
JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-20250131",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/hle ++generation_key=judgement ++add_generation_stats=False"
```


## How we extract answers

After answer generation by the `model`, by default we will extract the answer from the last `\boxed{}` field in the generated solution. This is consistent
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

## How we compare answers

By default all benchmarks in [Supported benchmarks](#supported-benchmarks) use
[generic/math](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) prompt config.

Most answers in these benchmarks can be compared using a
[symbolic checker](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/math_grader.py#L47)
but a few require using LLM-as-a-judge. By default those benchmarks will use GPT-4.1 and thus require OPENAI_API_KEY
to be defined. If you want to host a local judge model instead, you can change benchmark parameters, from the above command, like this:

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

## Custom prompt for LLM-as-Judge

If you want to use custom prompt for LLM-as-Judge, you can do so as following. In the benchmark definition (`__init__.py` script):

```bash
JUDGE_PIPELINE_ARGS = {
    "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "server_type": "openai",
    "server_address": "https://integrate.api.nvidia.com/v1"
}
JUDGE_ARGS = "++prompt_config=prompts/myjudge.yaml ++generation_key=judgement ++add_generation_stats=False ++system_message=\"detailed thinking off"
```

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

### beyond-aime

- Benchmark is defined in [`nemo_skills/dataset/beyond-aime/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/beyond-aime/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME).