# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) model as an example.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

!!! tip "Interactive Chat Interface"

    Besides the benchmark numbers shown below, you can also interactively chat with OpenMath models using our
    [chat interface](../../basics/chat_interface.md). This allows you to easily test both Chain-of-Thought (CoT) and
    Tool-Integrated Reasoning (TIR) modes with code execution in a user-friendly web UI.

!!! note

    For small benchmarks such as AIME24 and AIME25 (30 problems each) it is expected to see significant variation
    across different evaluation reruns. We've seen the difference as large as 6% even for results that are averaged
    across 64 generations. So please don't expect to see exactly the same numbers as presented in our paper, but
    they should be within 3-6% of reported results.

## Download models

Get the model from HF. E.g.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir OpenMath-Nemotron-1.5B
```

## Prepare evaluation data

```bash
ns prepare_data comp-math-24-25 hle
```

## Run CoT evaluations

```bash
ns eval \
    --cluster=local \
    --model=/workspace/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6

ns eval \
    --cluster=local \
    --model=/workspace/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=hle:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --split=math \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6
```

This will take a very long time unless you run on slurm cluster.
If running on slurm, you can set `--num_jobs` to a bigger number of -1 to run
each benchmark in a separate node. The number of GPUs need to match what you used
in the conversion command.

For comp-math-24-25 our symbolic checker is good enough, so we can see the results right away by running

```bash
ns summarize_results /workspace/openmath-nemotron-1.5b-eval-cot/eval-results/comp-math-24-25 --metric_type math --cluster local
```

For hle-math it's necessary to run LLM-as-a-judge step to get accurate evaluation results.
We used [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) which you
can run with the following command, assuming you have the model downloaded and converted locally
or on cluster.

```bash
ns generate \
    --generation_type=math_judge \
    --cluster=local \
    --model=/hf_models/Qwen2.5-32B-Instruct \
    --server_type=sglang \
    --server_gpus=4 \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle \
    --input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results/hle
```

Alternatively, you can use an API model like gpt-4o, but the results might be different.
You need to define `OPENAI_API_KEY` for the command below to work.

```bash
ns generate \
    --generation_type=math_judge \
    --cluster=local \
    --model=gpt-4o \
    --server_type=openai \
    --server_address=https://api.openai.com/v1 \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle \
    --input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results/hle
```

To print the metrics run

```bash
ns summarize_results /workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle --metric_type math --cluster local
```

This should print the metrics including both symbolic and judge evaluation.

## Run TIR evaluations

To get TIR evaluation numbers, replace the generation commands like this

```bash
ns eval \
    --cluster=local \
    --model=/workspace/OpenMath-Nemotron-1.5B \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --with_sandbox \
    ++code_tags=openmath \
    ++prompt_config=openmath/tir \
    ++use_completions_api=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true \
    ++server.code_execution.add_remaining_code_executions=true \
    ++total_code_executions_in_prompt=8
```

The only exception is for [OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle)
you should use the following options instead

```bash
ns eval \
    --cluster=local \
    --model=/workspace/OpenMath-Nemotron-14B-Kaggle \
    --server_type=sglang \
    --output_dir=/workspace/openmath-nemotron-14b-kaggle-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --with_sandbox \
    ++code_tags=openmath \
    ++prompt_config=generic/math \
    ++use_completions_api=True \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true
```

All other commands are the same as in the [CoT part](#run-cot-evaluations).


## Run GenSelect evaluations

Here is a sample command to run GenSelect evaluation:

```bash
ns genselect \
    --preprocess_args="++input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle" \
    --model=/trt_models/openmath-nemotron-1.5b \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot/self_genselect_hle \
    --cluster=local \
    --server_type=sglang \
    --server_gpus=1 \
    --num_random_seeds=64
```

The output folder will have three folders (apart from log folders):

1. `comparison_instances`: This is the folder where input instances for genselect are kept.

2. `comparison_judgment`: Output of GenSelect judgments.

3. `hle` / `math`: Folder with outputs based on GenSelect's judgments. If `dataset` is not specified in the command, we create a folder with the name `math`

To print the metrics run:

```bash
ns summarize_results \
  /workspace/openmath-nemotron-1.5b-eval-cot/self_genselect_hle/hle \
  --metric_type math \
  --cluster local
```
