# Model evaluation

!!! info

    This pipeline starting script is [nemo_skills/pipeline/eval.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/eval.py)

    All extra parameters are passed to [nemo_skills/inference/generate.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/inference/generate.py)


We support many popular benchmarks and it's easy to add new in the future. E.g. we support

- Math problem solving: hmmt_feb25, brumo25, aime24, aime25, omni-math (and many more)
- Formal proofs in Lean: minif2f, proofnet
- Coding skills: livecodebench, human-eval, mbpp
- Chat/instruction following: ifeval, arena-hard, mt-bench
- General knowledge: mmlu, mmlu-pro, gpqa
- Long context: ruler

See [nemo_skills/dataset](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset) where each folder is a benchmark we support.

Here is how to run evaluation (using API model as an example,
but same command works with self-hosted models both locally and on slurm).
Make sure that `/workspace` is mounted inside of your
[cluster config](../basics/cluster-configs.md).

## Preparing data

You need to run the following commands to prepare the data.

```bash
ns prepare_data
```

If you're only interested in a subset of datasets (e.g. only math-related or code-related), run with
`--dataset_groups ...` and if you only need a couple of specific datasets, list them directly e.g.

```bash
ns prepare_data gsm8k human-eval mmlu ifeval
```

If you have the repo cloned locally, the data files will be available inside `nemo_skills/dataset/<benchmark>/<split>.jsonl`
and if you installed from pip, they will be downloaded to wherever the repo is installed, which you can figure out by running

```bash
python -c "import nemo_skills; print(nemo_skills.__path__)"
```

Some benchmarks (e.g. ruler) require extra parameters to be passed to the prepare_data script. Thus you'd need to explicitly
call `ns prepare_data` for all of them, e.g. for ruler you can use

```bash
ns prepare_data ruler --setup=llama_128k --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct --max_seq_length=131072
```

## Greedy decoding

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=gsm8k,human-eval \
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
---------------------------------------- gsm8k ----------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1          | 1319        | 180        | 164         | 81.96%           | 4.93%


------------------------------------------- human-eval -------------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | passing_base_tests | passing_plus_tests
pass@1          | 164         | 199        | 29          | 64.63%             | 60.37%
```

The [summarize_results](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/summarize_results.py) script
will fetch the results from cluster automatically if you ran the job there.

!!! note

    The numbers above don't match reported numbers for Llama 3.1 because we are not using
    the same prompts by default. You would need to modify the prompt config for each specific benchmark
    to match the results exactly. E.g. to match gsm8k numbers add `++prompt_config=llama3/gsm8k`
    (but we didn't include all the prompts used for Llama3 evaluation, only a small subset as an example).

## Using multiple samples

You can add `:<num repeats>` after the benchmark name to repeat evaluation multiple times with high temperature
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
---------------------------------------- gsm8k -----------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-4] | 1319        | 180        | 680         | 80.44%           | 6.31%
majority@4       | 1319        | 180        | 680         | 88.40%           | 0.15%
pass@4           | 1319        | 180        | 680         | 93.63%           | 0.15%


-------------------------------------------- human-eval -------------------------------------------
evaluation_mode  | num_entries | avg_tokens | gen_seconds | passing_base_tests | passing_plus_tests
pass@1[avg-of-4] | 164         | 215        | 219         | 64.63%             | 59.30%
pass@4           | 164         | 215        | 219         | 79.27%             | 74.39%
```


## Using data on cluster

Some benchmarks (e.g. ruler) have very large input datasets and it's inefficient to prepare them on local machine and
keep uploading on cluster with every evaluation job. Instead, you can prepare them on cluster directly. To do that,
run prepare_data command with `--data_dir` and `--cluster` options, e.g.

```bash
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=slurm \
    ruler --setup llama_128k --tokenizer_path meta-llama/Llama-3.1-8B-Instruct --max_seq_length 130900
```

Then during evaluation, you'd need to provide the same `data_dir` argument and it will read the data from cluster
directly. You can also use `NEMO_SKILLS_DATA_DIR` environment variable instead of an explicit argument.

Here is an example evaluation command for ruler that uses data_dir parameter

```python
from nemo_skills.pipeline.cli import eval, wrap_arguments

eval(
    # using a low number of concurrent requests since it's almost entirely prefill stage
    ctx=wrap_arguments("++max_concurrent_requests=32"),
    cluster="slurm",
    model="/hf_models/Meta-Llama-3.1-8B-Instruct",
    server_type="sglang",
    output_dir="/workspace/eval-ruler",
    data_dir="/workspace/ns-data",
    benchmarks="ruler.llama_128k",
    server_gpus=8,
    expname="eval-ruler",
)
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
EVAL_ARGS = "++eval_type=math"
GENERATION_ARGS = ""
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