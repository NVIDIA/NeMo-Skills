---
date: 2025-08-15
readtime: 15
hide:
  - toc
---

# Reproducing Llama-Nemotron-Super-49B-V1.5 Evals

In this tutorial, we will reproduce the evals for the Llama-3.3-Nemotron-Super-49B-v1.5 model using NeMo-Skills.
For an introduction to the NeMo-Skills framework, we recommend going over [our introductory tutorial](../../basics/index.md).


We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly if running on slurm or using different paths.

<!-- more -->

## Download the model

Get the model from HF.
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 --local-dir /workspace/Llama-3_3-Nemotron-Super-49B-v1_5
```

!!!note
     In most cases, we can define `HF_HOME` in the cluster config to a mounted directory, and refer to models by their huggingface names such as `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` in this case. However, in this example, we download the model to an explicit location because we rely on the tool parsing script which is part of the huggingface repo. Alternatively, users can download the model to the `HF_HOME` and separately download the [tool parsing script](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py){target="_blank"} to another mounted location.

## Prepare evaluation data

We will evaluate the model on the following:

- Science & General reasoning benchmarks:
    - GPQA
    - MMLU-Pro
    - HLE

- Coding reasoning benchmarks
    - LiveCodeBench
    - SciCode

- Math reasoning benchmarks:
    - MATH-500
    - AIME24
    - AIME25

- Tool-calling:
    - BFCL v3


Here is the command to prepare these datasets using NeMo-Skills:

```bash
ns prepare_data gpqa mmlu-pro hle livecodebench scicode bfcl_v3 math-500 aime24 aime25
```


## Evaluation commands

Llama-3.3-Nemotron-Super-49B-v1.5 can perform inference in both reasoning on and off modes.
We detail the evaluation commands and results for both the modes.
Note that you might not get exactly the same numbers as reported here because of the stochastic nature of LLM generations.

!!! note
    The commands provided here assume you're working with a local machine where benchmarks/subsets are evaluated sequentially which will take a very long time. If running on slurm, by default we will run each benchmark and their random seeds as an independent job.



### Reasoning-on Evals

For the reasoning mode evals, we follow the recommended recipe of setting:

- temperature to 0.6
- top-p to 0.95
- system_message to empty i.e. ''
- maximum number of generated tokens to 65536

#### Command for Math, Code, and Science Reasoning Eval (Reasoning on)

The following command evaluates the model on GPQA, MMLU-Pro, Scicode, MATH-500, AIME24, and AIME25 across 16 different runs for all benchmarks. We have highlighted the inference settings recommended above in the following command:


```bash hl_lines="8-12"
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5/ \
    --benchmarks=gpqa:16,mmlu-pro:16,scicode:16,math-500:16,aime24:16,aime25:16 \
    --server_gpus=2 \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message=''
```

For LiveCodeBench, we additionally specify the exact split on which we evaluate the benchmark. In the following command, we evaluate the model on the 166 problems from the 1 October 2024 to 1 March 2025 subset from release_v5. To evaluate on the Artificial Analysis Index (AAI) split, set split to `test_v5_2407_2412`:

```bash hl_lines="7"
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5/ \
    --benchmarks=livecodebench:16 \
    --split=test_v5_2410_2502 \
    --server_gpus=2 \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message=''
```

#### Command for HLE Eval (Reasoning on)


For HLE, because symbolic comparison is not sufficient to determine the correctness of the output, we use the recommended `o3-mini-20250131` model as the judge. Note that this model is the default in NeMo-Skills, and we have just added this argument for illustration purposes. To evaluate for the [Artificial Analysis Index (AAI) setting, please use the gpt-4o-20240806 model as the judge](https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-overview){target="_blank"}.

Note that using any of the OpenAI hosted models requires `OPENAI_API_KEY`. Alternatively, a self-hosted judge model can also be used for judgement. For example, `--judge_model="/workspace/Llama-3_3-Nemotron-Super-49B-v1_5"`  in tandem with `--judge_server_type="vllm" --judge_server_gpus 2` will use the `Llama-3_3-Nemotron-Super-49B-v1_5` itself as a judge.


```bash hl_lines="8-9"
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5/ \
    --benchmarks=hle:16 \
    --server_gpus=2 \
    --judge_model="o3-mini-20250131" \
    --extra_judge_args="++inference.tokens_to_generate=4096 ++max_concurrent_requests=8" \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message=''
```

!!! note
    For Llama-Nemotron-Super-49B-V1.5, we found that the difference in judge models can result in almost 0.8-1% performance difference. Our earlier experiments with GPT-4.1 as the judge was giving a performance of 6.8%. This can explain why [AAI reports a performance of 6.8%](https://artificialanalysis.ai/models/llama-nemotron-super-49b-v1-5-reasoning#intelligence-evaluations){target="_blank"} vs our reproduced performance of 7.75%.

!!! note
    If the OpenAI API throws the `Rate limit exceeded` error, please reduce the `max_concurrent_requests` value in the `extra_judge_args` argument and restart the job.


#### Command for BFCL Eval (Reasoning on)

Tool-calling benchmarks require tool-call parsing and execution. NeMo-Skills supports both client-side parsing (default) and server-side parsing. For server-side parsing, the vLLM server requires the parsing details as highlighted in the below command:
```bash hl_lines="12-16"
ns eval \
    --cluster=local \
    --benchmarks=bfcl_v3 \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/ \
    --server_gpus=2 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_tool_calling/ \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='' \
    ++use_client_parsing=False \
    --server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                    --tool-call-parser \"llama_nemotron_json\" \
                    --enable-auto-tool-choice"
```



### Reasoning-on Results

The eval jobs also launch a dependent job to perform metrics calculation and store the result in a file called `metrics.json`.
In our running example, for a benchmark such as aime25, the `metrics.json` would be located at `/workspace/llama_nemotron_49b_1_5/eval-results/aime25/metrics.json`.
This metrics calculation is done typically by the `summarize_results` pipeline, except in the case of BFCL where the metrics are calculated by a BFCL specific script because BFCL has a specific way of combining subtask accuracy to obtain the overall accuracy.

To print the results for these benchmarks (except for BFCL), we could rerun the `summarize_results` script manually as follows:
```bash
ns summarize_results --cluster=local /workspace/llama_nemotron_49b_1_5/eval-results/{BENCHMARK}
```


#### Results for Science & General Reasoning benchmarks (Reasoning on)

```
------------------------------------------ gpqa -----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 198         | 11046      | 1986        | 74.65%           | 0.60%
majority@16       | 198         | 11046      | 1986        | 78.28%           | 0.00%
pass@16           | 198         | 11046      | 1986        | 92.93%           | 0.00%

---------------------------------------- mmlu-pro ---------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 12032       | 4879       | 12516       | 81.44%           | 0.05%
majority@16       | 12032       | 4879       | 12516       | 83.05%           | 0.00%
pass@16           | 12032       | 4879       | 12516       | 91.32%           | 0.00%

-------------------------------------------------- hle --------------------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | judge_correct | symbolic_correct | no_answer
pass@1[avg-of-16] | 2158        | 12111      | 7782        | 7.75%         | 2.40%            | 64.13%
majority@16       | 2158        | 12111      | 7782        | 4.31%         | 3.43%            | 49.91%
pass@16           | 2158        | 12111      | 7782        | 27.80%        | 10.10%           | 49.91%
```

!!!note
    The `majority` metric for most reasoning benchmarks typically improves over the corresponding `pass@1` numbers. For HLE, the `majority` number is lower than `pass@1` which can be counterintuitive but it has to with our metric calculation logic. For HLE, the final answer is contained in the generated solution but it is not easily extractable by rule-based systems as in the case of math where the model is instructed to put the final answer in \boxed{}. Thus, for certain questions the `predicted_answer` field is null but the LLM-as-a-judge is still able to evaluate the generated solution. The majority metric performs clustering over `predicted_answer` which currently incorrectly removes from consideration some of the correct solutions for which the `predicted_answer` is None.


#### Results for Code Reasoning benchmarks (Reasoning on)

```
--------------------------- livecodebench ---------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | accuracy
pass@1[avg-of-16] | 166         | 18881      | 1552        | 71.72%
pass@16           | 166         | 18881      | 1552        | 87.35%

--------------------------------------------------- scicode ----------------------------------------------------
evaluation_mode   | avg_tokens | gen_seconds | problem_accuracy | subtask_accuracy | num_problems | num_subtasks
pass@1[avg-of-16] | 43481      | 69963       | 3.08%            | 28.91%           | 65           | 288
pass@16           | 43481      | 69963       | 7.69%            | 40.97%           | 65           | 288
```

#### Results for Math Reasoning benchmarks (Reasoning on)

```
---------------------------------------- math-500 ---------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 500         | 5807       | 2828        | 97.79%           | 0.28%
majority@16       | 500         | 5807       | 2828        | 99.00%           | 0.00%
pass@16           | 500         | 5807       | 2828        | 99.40%           | 0.00%


----------------------------------------- aime24 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 19875      | 2042        | 88.54%           | 1.88%
majority@16       | 30          | 19875      | 2042        | 93.33%           | 0.00%
pass@16           | 30          | 19875      | 2042        | 93.33%           | 0.00%


----------------------------------------- aime25 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 23366      | 832         | 84.38%           | 3.96%
majority@16       | 30          | 23366      | 832         | 93.33%           | 0.00%
pass@16           | 30          | 23366      | 832         | 93.33%           | 0.00%
```


#### Results for Tool Calling  (Reasoning on)

```
----------------------- bfcl_v3 ------------------------
| Category                    | num_entries | accuracy |
|-----------------------------|-------------|----------|
| overall_accuracy            | 4441        | 72.64%   |
| overall_non_live            | 1390        | 88.20%   |
| non_live_ast                | 1150        | 88.58%   |
| irrelevance                 | 240         | 86.67%   |
| overall_live                | 2251        | 83.34%   |
| live_ast                    | 1351        | 82.68%   |
| live_irrelevance            | 882         | 84.47%   |
| live_relevance              | 18          | 77.78%   |
| overall_multi_turn          | 800         | 46.38%   |

```

!!! note
    Currently `summarize_results` doesn't support benchmarks like BFCL v3 which have their specific logic of combining subset scores to arrive at the overall score. This table was created by formatting the `metrics.json` file from `/workspace/llama_nemotron_49b_1_5_tool_calling/bfcl_v3/metrics.json`.



### Reasoning-off Evals

For the non-reasoning mode evals, we follow the recommended recipe of setting:

- temperature to 0.0
- top-p to 1.0
- system_message to '/no_think'
- keep the maximum number of generated tokens to 65536

#### Command for Math, Code, and Science Reasoning Eval (Reasoning off)

The following command evaluates the model on GPQA, MMLU-Pro, Scicode, MATH-500, AIME24, and AIME25 across 16 different runs for all benchmarks. We have highlighted the inference settings recommended above in the following command:


```bash hl_lines="9-12"
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_reasoning_off/ \
    --benchmarks=gpqa:16,mmlu-pro:16,scicode:16,math-500:16,aime24:16,aime25:16 \
    --server_gpus=2 \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.0 \
    ++inference.top_p=1.0 \
    ++system_message='/no_think'
```

For LiveCodeBench, the command is:

```bash
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_reasoning_off/ \
    --benchmarks=livecodebench:16 \
    --split=test_v5_2410_2502 \
    --server_gpus=2 \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.0 \
    ++inference.top_p=1.0 \
    ++system_message='/no_think'
```

#### Command for HLE Eval (Reasoning off)


```bash
ns eval \
    --cluster=local \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_reasoning_off/ \
    --benchmarks=hle:16 \
    --server_gpus=2 \
    --judge_model="o3-mini-20250131" \
    --extra_judge_args="++inference.tokens_to_generate=4096 ++max_concurrent_requests=8" \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.0 \
    ++inference.top_p=1.0 \
    ++system_message='/no_think'
```

#### Command for BFCL Eval (Reasoning off)

```bash
ns eval \
    --cluster=local \
    --benchmarks=bfcl_v3 \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/ \
    --server_gpus=2 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_reasoning_off_tool_calling/ \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.0 \
    ++inference.top_p=1.0 \
    ++system_message='/no_think' \
    ++use_client_parsing=False \
    --server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                   --tool-call-parser \"llama_nemotron_json\" \
                   --enable-auto-tool-choice"
```


### Reasoning-off Results


We use the `summarize_results` on the reasoning_off results directory as follows:

```bash
ns summarize_results --cluster=local /workspace/llama_nemotron_49b_1_5_reasoning_off/eval-results/{BENCHMARK}
```


#### Results for Science & General Reasoning benchmarks (Reasoning off)

```
------------------------------------------ gpqa -----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 198         | 853        | 1552        | 51.61%           | 0.25%
majority@16       | 198         | 853        | 1552        | 52.53%           | 0.00%
pass@16           | 198         | 853        | 1552        | 74.75%           | 0.00%

---------------------------------------- mmlu-pro ---------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 12032       | 625        | 5684        | 69.19%           | 0.34%
majority@16       | 12032       | 625        | 5684        | 69.94%           | 0.01%
pass@16           | 12032       | 625        | 5684        | 77.67%           | 0.01%

-------------------------------------------------- hle --------------------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | judge_correct | symbolic_correct | no_answer
pass@1[avg-of-16] | 2158        | 1349       | 2667        | 3.92%         | 1.30%            | 59.09%
majority@16       | 2158        | 1349       | 2667        | 1.53%         | 1.44%            | 47.03%
pass@16           | 2158        | 1349       | 2667        | 12.09%        | 3.29%            | 47.03%
```


#### Results for Code Reasoning benchmarks (Reasoning off)

```
--------------------------- livecodebench ---------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | accuracy
pass@1[avg-of-16] | 166         | 609        | 1156        | 29.89%
pass@16           | 166         | 609        | 1156        | 33.73%

--------------------------------------------------- scicode ----------------------------------------------------
evaluation_mode   | avg_tokens | gen_seconds | problem_accuracy | subtask_accuracy | num_problems | num_subtasks
pass@1[avg-of-16] | 3067       | 66547       | 0.00%            | 19.44%           | 65           | 288
pass@16           | 3067       | 66547       | 0.00%            | 29.51%           | 65           | 288
```

#### Results for Math Reasoning benchmarks (Reasoning off)

```
---------------------------------------- math-500 ---------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 500         | 765        | 1185        | 75.55%           | 0.26%
majority@16       | 500         | 765        | 1185        | 76.00%           | 0.00%
pass@16           | 500         | 765        | 1185        | 84.00%           | 0.00%

----------------------------------------- aime24 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 3611       | 1165        | 16.88%           | 3.75%
majority@16       | 30          | 3611       | 1165        | 16.67%           | 0.00%
pass@16           | 30          | 3611       | 1165        | 33.33%           | 0.00%

----------------------------------------- aime25 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-16] | 30          | 1720       | 1149        | 5.42%            | 1.25%
majority@16       | 30          | 1720       | 1149        | 6.67%            | 0.00%
pass@16           | 30          | 1720       | 1149        | 10.00%           | 0.00%
```

#### Results for Tool Calling  (Reasoning off)


```
----------------------- bfcl_v3 ------------------------
| Category                    | num_entries | accuracy |
|-----------------------------|-------------|----------|
| overall_accuracy            | 4441        | 68.52%   |
| overall_non_live            | 1390        | 87.55%   |
| non_live_ast                | 1150        | 87.35%   |
| irrelevance                 | 240         | 88.33%   |
| overall_live                | 2251        | 81.87%   |
| live_ast                    | 1351        | 79.79%   |
| live_irrelevance            | 882         | 85.60%   |
| live_relevance              | 18          | 55.56%   |
| overall_multi_turn          | 800         | 36.13%   |
```

The reasoning-on vs reasoning-off comparison shows inference-time scaling's impact: higher accuracy at the cost of more tokens and longer generation times.
