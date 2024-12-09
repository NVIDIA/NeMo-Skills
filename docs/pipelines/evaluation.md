# Model evaluation

!!! info

    This pipeline starting script is [nemo_skills/pipeline/eval.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/eval.py)

    All extra parameters are passed to [nemo_skills/inference/generate.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/inference/generate.py)


We support many popular benchmarks and it's easy to add new in the future. E.g. we support

- Math problem solving: gsm8k, math, amc23, aime24 (and many more)
- Coding skills: human-eval, mbpp
- Chat/instruction following: ifeval, arena-hard
- General knowledge: mmlu (generative)

See [nemo_skills/dataset](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset) where each folder is a benchmark we support.

Here is how to run evaluation (using API model as an example,
but same command works with self-hosted models both locally and on slurm).
Make sure that `/workspace` is mounted inside of your
[cluster config](../basics/prerequisites.md#cluster-configs).

## Greedy decoding

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=gsm8k:0,human-eval:0 \
    --output_dir=/workspace/test-eval
```

This will run evaluation on gsm8k and human-eval for Llama 3.1 8B model. If you're running
on slurm by default each benchmark is run in a separate job, but you can control this with
`--num_jobs` parameter.

After the evaluation is done, you can get metrics by calling

```bash
ns summarize_results --cluster local /workspace/test-eval
```

Which should print the following

```
------------------------- gsm8k -------------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
greedy          | 1319        | 82.34         | 0.91


------------------------------ human-eval -----------------------------
evaluation_mode | num_entries | passing_base_tests | passing_plus_tests
greedy          | 164         | 67.68              | 62.20
```

The [summarize_results](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/summarize_results.py) script
will fetch the results from cluster automatically if you ran the job there.

!!! note

    The numbers above don't match reported numbers for Llama 3.1 because we are not using
    the same prompts by default. You would need to modify the prompt config for each specific benchmark
    to match the results exactly. E.g. to match gsm8k numbers add `++prompt_config=llama3/gsm8k`
    (but we didn't include all the prompts used for Llama3 evaluation, only a small subset as an example).

## Using multiple samples

The `:0` part after benchmark name means that we only want to run
greedy decoding, but if you set `:4` it will run greedy + 4 samples with high temperature
that can be used for majority voting or estimating pass@k. E.g. if we run with

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks gsm8k:4,human-eval:4 \
    --output_dir=/workspace/test-eval
```

you will see the following output after summarizing results

```
-------------------------- gsm8k ---------------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
greedy          | 1319        | 82.34            | 0.91
majority@4      | 1319        | 87.95            | 0.00
pass@4          | 1319        | 93.78            | 0.00


------------------------------ human-eval -----------------------------
evaluation_mode | num_entries | passing_base_tests | passing_plus_tests
greedy          | 164         | 67.68              | 62.20
pass@4          | 164         | 78.66              | 72.56
```

## How the benchmarks are defined

Each benchmark exists as a separate folder inside
[nemo_skills/dataset](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset). Inside
those folders there needs to be `prepare.py` script which can be run to download and format benchmark
data into a .jsonl input file (or files if it supports train/validation besides a test split) that
our scripts can understand. There also needs to be an `__init__.py` that defines some default variables
for that benchmark, such as prompt config, evaluation type, metrics class and a few more.

This information is than used inside eval pipeline to initialize default setup (but all arguments can
be changed from the command line).

Let's look at gsm8k to understand a bit more how each part of the evaluation works.

Inside [nemo_skills/dataset/gsm8k/\_\_init\_\_.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/gsm8k/__init__.py) we see the following

```python

# settings that define how evaluation should be done by default (all can be changed from cmdline)
PROMPT_CONFIG = 'generic/math'
DATASET_GROUP = 'math'
METRICS_TYPE = "math"
DEFAULT_EVAL_ARGS = "++eval_type=math"
DEFAULT_GENERATION_ARGS = ""
```

The prompt config and default generation arguments are passed to the
[nemo_skills/inference/generate.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/inference/generate.py) and
the default eval args are passed to the
[nemo_skills/evaluation/evaluate_results.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/evaluate_results.py).
The dataset group is used by [nemo_skills/dataset/prepare.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/prepare.py)
to help download only benchmarks from a particular group if `--dataset_groups` parameter is used.
Finally, the metrics class is used by [nemo_skills/evaluation/metrics.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/metrics.py)
which is called when you run summarize results pipeline.

To create a new benchmark in most cases you only need to add a new prepare script and the corresponding
default prompt. If the new benchmark needs some not-supported post-processing or metric summarization
you'd need to also add a new evaluation type and a new metrics class.