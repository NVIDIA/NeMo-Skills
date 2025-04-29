# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) model as an example.
We assume you have `/workspace` defined in your [cluster config](../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

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

## Convert to TensorRT-LLM

Convert the model to TensorRT-LLM format. This is optional, but highly recommended for more exact
results and faster inference. If you skip it, replace `--server_type trtllm` with `--server-type vllm` (or sglang)
in the commands below and change model path to `/workspace/OpenMath-Nemotron-1.5B`.

```bash
ns convert \
    --cluster=local \
    --input_model=/workspace/OpenMath-Nemotron-1.5B \
    --output_model=/workspace/openmath-nemotron-1.5b-trtllm \
    --convert_from=hf \
    --convert_to=trtllm \
    --model_type=qwen \
    --num_gpus=1 \
    --hf_model_name=nvidia/OpenMath-Nemotron-1.5B \
    --max_input_len 50000 \
    --max_seq_len 50000
```

We are converted with longer length since HLE-math benchmark has a few very long prompts.
You can change the number of GPUs if you have more than 1, but don't use more than 4 for 1.5B and 7B models.

## Prepare evaluation data

```bash
python -m nemo_skills.dataset.prepare comp-math-24-25 hle
```

## Run CoT evaluations

```bash
ns eval \
    --cluster=local \
    --model=/workspace/openmath-nemotron-1.5b-trtllm \
    --server_type=trtllm \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --skip_greedy \
    ++prompt_template=qwen-instruct \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6

ns eval \
    --cluster=local \
    --model=/workspace/openmath-nemotron-1.5b-trtllm \
    --server_type=trtllm \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot \
    --benchmarks=hle:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --skip_greedy \
    --split=math \
    ++prompt_template=qwen-instruct \
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

```
------------------- comp-math-24-25-all --------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
majority@64     | 256         | 58.20%           | 0.00%
pass@64         | 256         | 75.39%           | 0.00%
pass@1[64]      | 256         | 44.59%           | 0.00%


------------------ comp-math-24-25-aime25 ------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
majority@64     | 30          | 66.67%           | 0.00%
pass@64         | 30          | 83.33%           | 0.00%
pass@1[64]      | 30          | 50.52%           | 0.00%


------------------ comp-math-24-25-aime24 ------------------
evaluation_mode | num_entries | symbolic_correct | no_answer
majority@64     | 30          | 80.00%           | 0.00%
pass@64         | 30          | 80.00%           | 0.00%
pass@1[64]      | 30          | 62.60%           | 0.00%


---------------- comp-math-24-25-hmmt-24-25 ----------------
evaluation_mode | num_entries | symbolic_correct | no_answer
majority@64     | 196         | 53.57%           | 0.00%
pass@64         | 196         | 73.47%           | 0.00%
pass@1[64]      | 196         | 40.92%           | 0.00%
```

For hle-math it's necessary to run LLM-as-a-judge step to get accurate evaluation results.
We used [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) which you
can run with the following command, assuming you have the model downloaded and converted locally
or on cluster.

```bash
ns generate \
    --generation_type=math_judge \
    --cluster=local \
    --model=/trt_models/qwen2.5-32b-instruct \
    --server_type=trtllm \
    --server_gpus=4 \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle \
    ++input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results/hle
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
    ++input_dir=/workspace/openmath-nemotron-1.5b-eval-cot/eval-results/hle
done
```

To print the metrics run

```bash
ns summarize_results /workspace/openmath-nemotron-1.5b-eval-cot/eval-results-judged/hle --metric_type math --cluster local
```

This should print the metrics including both symbolic and judge evaluation.

```
------------------------------------------------ hle -----------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
majority@64     | 975         | 0.82%            | 5.41%         | 0.72%        | 5.41%       | 0.00%
pass@64         | 975         | 14.05%           | 38.36%        | 13.85%       | 38.56%      | 0.00%
pass@1[64]      | 975         | 1.18%            | 5.41%         | 3.06%        | 3.53%       | 0.00%
```

## Run TIR evaluations

To get TIR evaluation numbers, replace the generation commands like this

```bash
ns eval \
    --cluster=local \
    --model=/workspace/openmath-nemotron-1.5b-trtllm \
    --server_type=trtllm \
    --output_dir=/workspace/openmath-nemotron-1.5b-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --skip_greedy \
    ++prompt_template=openmath-instruct \
    ++prompt_config=openmath/tir \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true \
    ++server.code_execution.add_remaining_code_executions=true
```

The only exception is for [OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle)
you should use the following options instead

```bash
ns eval \
    --cluster=local \
    --model=/workspace/openmath-nemotron-14b-kaggle-trtllm \
    --server_type=trtllm \
    --output_dir=/workspace/openmath-nemotron-14b-kaggle-eval-tir \
    --benchmarks=comp-math-24-25:64 \
    --server_gpus=1 \
    --num_jobs=1 \
    --skip_greedy \
    ++prompt_template=openmath-instruct \
    ++prompt_config=generic/math \
    ++inference.tokens_to_generate=32768 \
    ++inference.temperature=0.6 \
    ++code_execution=true
```

All other commands are the same as in the [CoT part](#run-cot-evaluations).


## Run GenSelect evaluations

Coming soon!