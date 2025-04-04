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

We also provide 10 example subset of the raw data in [configs/example-data.txt](/recipes/omr1/configs/example-data.txt).
If you want to test the pipeline but don't have any GPUs, you can use the scripts below with `--config demo` flag
and it will run all steps on those 10 examples using [Nvidia NIM models](https://build.nvidia.com/). Make sure to define
`NVIDIA_API_KEY` environment variable for this to work.

## Problem generation pipeline

