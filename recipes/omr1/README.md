# OpenMathReasoning-1

OpenMathReasoning-1 dataset consists of mathematical problems collected from [AoPS community forums](https://artofproblemsolving.com/community).
Here are the steps to reproduce the dataset creation process.

## Data scraping

There is a great open-source [AoPS-Instruct repository](https://github.com/dsl-lab/aops) where you can find the scripts to scrape
the data. There is also a [DeepStudentLlama/AoPS-Instruct HF dataset](DeepStudentLlama/AoPS-Instruct) where the raw forum data can be found.
While we didn't use that repository/dataset in our work directly, it should produce a similar output to our internal scripts.

To download and preprocess raw data you can run

```bash
python scripts/prepare_raw_data.py
```

This script will rename certain columns in the original dataset to align with our scripts, combine forum discussions into
a single string, remove quotes and truncate the discussions that are longer than 24000 tokens. The prepared data will be
saved as `raw_aops_data.jsonl`.

The output file should have ~550k rows, so all of the following commands will take a very long time and require a big
number of GPUs if you want to run them on full data. If you just want to try out the pipeline, we recommend to subsample
the dataset by e.g. running

```bash
mv raw_aops_data.jsonl raw_aops_data_full.jsonl && head -n 1000 raw_aops_data_full.jsonl > raw_aops_data.jsonl
```

## Problem generation

For all the following commands we assume that your data is available under `/workspace/omr1-recipe/raw_aops_data.jsonl`.
`/workspace` should be mounted in your [cluster config](/docs/basics/prerequisites.md#cluster-configs) and data uploaded
on cluster if you're not running locally. We also assume that you have `/trt_models` mounted in your cluster config and
[Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) is available at `/trt_models/qwen2.5-32b-instruct`.
You can follow instructions in [checkpoint conversion](/docs/pipelines/checkpoint-conversion.md) to learn how to build
TensorRT-LLM checkpoint or change code in `pipeline.py` to replace server with `vllm` or `sglang`.

You can override all of the above through the arguments of `scripts/pipeline.py`.

### Problem extraction

