# Long-context

More details are coming soon!

## Supported benchmarks

### ruler

- Benchmark is defined in [`nemo_skills/dataset/ruler/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/ruler/__init__.py)
- Original benchmark source is [here](https://github.com/NVIDIA/RULER).

### mrcr

- Benchmark is defined in [`nemo_skills/dataset/mrcr/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mrcr/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/openai/mrcr).

### aalcr
- Benchmark is defined in [`nemo_skills/dataset/aalcr/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/aalcr/__init__.py)
- Original benchmark source is [here] (https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) and the reported scores by AA is here [here] (https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning).

#### Data preparation.
```bash
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=<cluster_config> \
    aalcr
```
You can also prepare a subset of the data with limited context window.
```bash
    --max_context_window 100000 --setup test_100k
```
#### Running evaluation.
It follows official AA-LCR implementation. Qwen3-235B-A22B-Instruct-2507 is served as judge and evaluation is conducted four time.
```bash
model=Qwen2.5-7B-Instruct-1M
ns eval \
    --cluster=<cluster_config> \
    --data_dir=/workspace/ns-data \
    --server_gpus=8 \
    --server_type=sglang \
    --model=/hf_models/$model \
    --benchmarks=aalcr:4 \
    --output_dir=/workspace/aalcr/$split/$model \
    --judge_pipeline_args='model=/hf_models/Qwen3-235B-A22B-Instruct-2507 server_type=sglang server_gpus=8 ' \
    --judget_args='++prompt_config=judge/aalcr ++generation_key=judgement ++add_generation_stats=False ' \
    --server_args='--disable-cuda-graph' \
```