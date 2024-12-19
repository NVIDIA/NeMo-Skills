# Dataset construction

Here are the commands you can run to re-create [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
We assume you have `/workspace` defined in your [cluster config](../basics/prerequisites.md#cluster-configs) and are running
all commands on a slurm cluster. Change the commands accordingly if running locally
(but it's going to take a lot of time).
We also assume you have the [Llama3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
on that cluster inside `/trt_models/llama-3.1-405b-instruct` (should be mounted in your config)
that's been [converted](../pipelines/checkpoint-conversion.md) to TensorRT-LLM format.
See [generation docs](../pipelines/generation.md) for how you can change the below commands to instead
run inference through Nvidia NIM API.

## Prepare the seed data

```bash
python -m nemo_skills.dataset.prepare gsm8k math
```

## Solution augmentation
We generate multiple new solutions for each of the original training set problems.

MATH dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/trt_models/llama-3.1-405b-instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=512 \
    --output_dir=/workspace/solution-augmentation/math \
    --eval_args="++eval_type=math" \
    ++dataset=math \
    ++split=train_full \
    ++prompt_config=generic/math-base \
    ++examples_type=math_text_detailed \
    ++prompt_template=llama3-base
```

GSM8K dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/trt_models/llama-3.1-405b-instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=64 \
    --output_dir=/workspace/solution-augmentation/gsm8k \
    --eval_args="++eval_type=math" \
    ++dataset=gsm8k \
    ++split=train_full \
    ++prompt_config=generic/math-base \
    ++examples_type=gsm8k_text_detailed \
    ++prompt_template=llama3-base
```

## Problem augmentation
We generate new problems using the problems from the training sets as a "seed".

MATH dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/trt_models/llama-3.1-405b-instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=80 \
    --output_dir=/workspace/problem-augmentation/math \
    ++dataset=math \
    ++split=train_full \
    ++prompt_config=generic/problem-augmentation \
    ++examples_type=math_problem_augmentation \
    ++prompt_template=llama3-instruct \
    ++generation_key=problem
```

GSM8K dataset.

```bash
ns generate \
    --cluster=slurm \
    --server_type=trtllm \
    --model=/trt_models/llama-3.1-405b-instruct \
    --server_gpus=8 \
    --server_nodes=2 \
    --num_random_seeds=10 \
    --output_dir=/workspace/problem-augmentation/gsm8k \
    ++dataset=gsm8k \
    ++split=train_full \
    ++prompt_config=generic/problem-augmentation-similar \
    ++examples_type=gsm8k_problem_augmentation \
    ++prompt_template=llama3-instruct \
    ++generation_key=problem
```

## Solutions for augmented data

Solution augmentation for the newly generated problems.
We generate 32 solutions for each of the new problems.

We use the Python API in commands below.

MATH dataset.

```python
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import generate

# we generated 80 new problems from each original seed problem, so we have a loop
# to now generate 32 solutions for each of those 80 new data files
for i in range(80):
    generate(
        cluster="slurm",
        server_type="trtllm",
        model="/trt_models/llama-3.1-405b-instruct",
        server_gpus=8,
        server_nodes=2,
        num_random_seeds=32,
        output_dir=f"/workspace/new-problems-solution-augmentation/math/problem-set{i}",
        ctx=wrap_arguments(
            f"++input_file=/workspace/solution-augmentation/math/generation/output-rs{i} "
            f"++prompt_config=generic/math-base "
            f"++examples_type=math_text_detailed "
            f"++prompt_template=llama3-base "
        ),
    )
```

GSM8K dataset.

```python
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import generate

# we generated 10 new problems from each original seed problem, so we have a loop
# to now generate 32 solutions for each of those 10 new data files
for i in range(10):
    generate(
        cluster="slurm",
        server_type="trtllm",
        model="/trt_models/llama-3.1-405b-instruct",
        server_gpus=8,
        server_nodes=2,
        num_random_seeds=32,
        output_dir=f"/workspace/new-problems-solution-augmentation/gsm8k/problem-set{i}",
        ctx=wrap_arguments(
            f"++input_file=/workspace/solution-augmentation/gsm8k/generation/output-rs{i} "
            f"++prompt_config=generic/math-base "
            f"++examples_type=gsm8k_text_detailed "
            f"++prompt_template=llama3-base "
        ),
    )
```

Add majority answer as the ground-truth answer.
Either copy the data locally or run this command on a slurm node.
You also need to specify the full path to where `/workspace` is mounted
(we will make it more convenient in the near future by providing the same
Python/cmdline API as for other scripts).

```python
import subprocess

# for MATH
data_folder = "<path to where /workspace is>/new-problems-solution-augmentation/math"
for i in range(80):
    cmd = (
        f'python -m nemo_skills.evaluation.fill_majority_answer '
        f'    ++input_files="{data_folder}/problem-set{i}/generation/output-rs*.jsonl" '
    )
    subprocess.run(cmd, shell=True, check=True)

# for GSM8K
data_folder = "<path to where /workspace is>/new-problems-solution-augmentation/gsm8k"
for i in range(10):
    cmd = (
        f'python -m nemo_skills.evaluation.fill_majority_answer '
        f'    ++input_files="{data_folder}/problem-set{i}/generation/output-rs*.jsonl" '
    )
    subprocess.run(cmd, shell=True, check=True)
```


## Decontamination
We test against GSM8K, MATH, AMC 2023, and AIME 2024.

Retrieve top-5 similar items from the test sets
```bash
python -m nemo_skills.inference.retrieve_similar \
    ++retrieve_from="./nemo_skills/dataset/gsm8k/test.jsonl ./nemo_skills/dataset/math/test.jsonl ./nemo_skills/dataset/amc23/test.jsonl ./nemo_skills/dataset/aime24/test.jsonl" \
    ++compare_to="<path to workspace>/new-problems-solution-augmentation/**/output-rs0.jsonl" \
    ++output_file=<path to workspace>/new-problems-solution-augmentation/contamination-retrieved.jsonl \
    ++top_k=5
```
!!! note

    Currently the above command doesn't run inside docker, so you will need to install additional packages.

Next, you need to run LLM inference to check those closest found problems from the output file. We use the Llama3.1-405B-Instruct model for this, and here's one way of doing it via Nvidia API catalog.

```bash
ns check_contamination \
    --cluster=local \
    --input_file=/workspace/new-problems-solution-augmentation/contamination-retrieved.jsonl \
    --output_file=/workspace/new-problems-solution-augmentation/contamination-llm.jsonl \
    --server_type=openai \
    --model=meta/llama-3.1-405b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    ++check_both_ways=True
```


## Converting to SFT format

Now all the data is generated and you can follow up by converting it to the SFT format.
We remove the problems marked as contaminated.
We also remove problems and solutions with length > 1024 Llama tokens.
To avoid the models from generating extremely short solutions, we remove solutions shorter than 200 characters.

```bash
python -m nemo_skills.training.prepare_sft_data \
    ++prompt_template=llama3-instruct \
    ++prompt_config=generic/math \
    ++input_files="<path to workspace>/solution-augmentation/**/output-rs*.jsonl <path to workspace>/new-problems-solution-augmentation/**/output-rs*.jsonl" \
    ++output_path=<path to workspace>/sft_data.jsonl \
    ++filters.remove_len_outlier_problems=true \
    ++max_problem_length=1024 \
    ++filters.remove_len_outlier_solutions=true \
    ++use_chars_for_min_length=true \
    ++min_solution_length=200 \
    ++hf_model_name="meta-llama/Meta-Llama-3.1-8B" \
    ++max_solution_length=1024 \
    ++filters.remove_contaminated=true \
    ++contamination_file=<path to workspace>/new-problems-solution-augmentation/contamination-llm.jsonl
```

## Dataset contamination explorer

To reproduce our dataset contamination explorer demo refer to [dataset_explorer_demo/README.md](https://github.com/NVIDIA/NeMo-Skills/blob/main/dataset_explorer_demo/README.md)