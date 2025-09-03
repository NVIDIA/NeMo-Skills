# Multilingual

Our multilingual benchmarks cover things like multilingual reasoning as well as machine translation (to be added).

All benchmarks in this category will have an extra `--language` argument with its associated `ns prepare` command, which allows you to choose which language(s) of the benchmark to run.
Once prepared, the `ns eval` command will run on all languages prepared, and the summarized results generated with `ns eval` will include per-language breakdowns.

## Supported benchmarks

### mmlu-prox

- Benchmark is defined in [`nemo_skills/dataset/mmlu-pro/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mmlu-prox/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/li-lab/MMLU-ProX).

Our evaluation template and answer extraction mechanism tries to match the configration in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu_prox).
Some reference numbers for reference and commands for reproduction:

| Model              | Type   |   en | de   | es   | fr   | it   | ja   |
|:-------------------|:-------|-----:|:-----|:-----|:-----|:-----|:-----|
| gpt-oss-120b       | Public       | 80.8 | -    | -    | -    | -    | -    |
| gpt-oss-120b       | Nemo-Skills  | 75.5 | 71.8 | 73.4 | 70.9 | 71.7 | 66.7 |
| mistral-3.1-small  | Public       | 62   | 58.5 | 59.4 | 60.6 | 59.6 | 54.4 |
| mistral-3.1-small  | Nemo-Skills  | 67.6 | 59.9 | 63.7 | 63.2 | 63.6 | 56.6 |
| qwen3-32b-thinking | Public       | 74.9 | 71.7 | 72.8 | 72.1 | 73.5 | 70.2 |
| qwen3-32b-thinking | Nemo-Skills  | 72.7 | 70.4 | 74.0 | 73.7 | 76.3 | 73.9 |

=== "GPT-OSS-120B"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=openai/gpt-oss-120b \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --num_chunks=16 \
        --server_type=vllm \
        --server_gpus=4 \
        --server_args='--async-scheduling' \
        ++inference.tokens_to_generate=2048
    ```

=== "Mistral-Small-3.1"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --server_type=vllm \
        --num_chunks=16 \
        --server_gpus=2 \
        --server_args='--tokenizer-mode mistral --config-format mistral --load-format mistral' \
        ++inference.tokens_to_generate=2048
    ```

=== "Qwen3-32B-Thinking"

    ```bash
    ns eval \
        --cluster=[cluster] \
        --model=Qwen/Qwen3-32B \
        --benchmarks mmlu-prox \
        --output_dir=[output dir] \
        --server_type=vllm \
        --num_chunks=32 \
        --server_gpus=2 \
        ++inference.temperature=0.6 \
        ++inference.top_k=20 \
        ++inference.tokens_to_generate=38912
    ```