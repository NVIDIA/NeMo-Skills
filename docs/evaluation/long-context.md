# Long-context

More details are coming soon!

## Supported benchmarks

### ruler

- Benchmark is defined in [`nemo_skills/dataset/ruler/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/ruler/__init__.py)
- Original benchmark source is [here](https://github.com/NVIDIA/RULER).

### mrcr

- Benchmark is defined in [`nemo_skills/dataset/mrcr/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/mrcr/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/openai/mrcr).

### [aalcr](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR)
- Benchmark is defined in [`nemo_skills/dataset/aalcr/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/aalcr/__init__.py)
- Original benchmark source is [here](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning).

Data preparation. You will need to get txt files using data_source_url or consult with AA.
```bash
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=<cluster_config> \
    aalcr --txt_file_folder=/workspace/do_not_share_data/lcr
```

You can also prepare a subset of the data with limited context window.
```bash
    --max_context_window 100000 --setup aalcr_100k
```

Example command for running evaluation. It follows official AA-LCR implementation. Qwen3-235B-A22B-Instruct-2507 is served as judge.

```bash
model=Qwen2.5-7B-Instruct-1M
ns eval \
    --cluster=<cluster_config> \
    --data_dir=/workspace/ns-data \
    --server_gpus=8 \
    --server_type=sglang \
    --model=/hf_models/$model \
    --benchmarks=aalcr:0 \
    --split=aalcr \
    --output_dir=/workspace/aalcr/$split/$model \
    (--server_args='--disable-cuda-graph')
```
