# Dataset construction

OpenMathReasoning-1 dataset consists of mathematical problems collected from [AoPS community forums](https://artofproblemsolving.com/community). Below we describe the pipeline used to create this dataset. All relevant scripts are available in
[recipes/omr1](/recipes/omr1) folder.

If you don't have a slurm cluster with a large number of GPUs,
you can still try out all the steps of our pipeline by using [Nvidia NIM models](https://build.nvidia.com/). We include
a 10-sample subset of the raw data in [configs/example-data.txt](/recipes/omr1/configs/example-data.txt) and you can
switch to that data and NIM models by adding `--mode demo` to all the pipeline commands. We also use different models
in this "demo" mode to make it faster, but you can change [configs/demo.yaml](/recipes/omr1/configs/demo.yaml) to pick
any other models supported in https://build.nvidia.com. Make sure to define `NVIDIA_API_KEY` environment variable for this to work
(and ignore scraping and model preparation steps as they are not needed when using NIM models).

Finally, please make sure to follow all necessary setup steps from the
[prerequisites documentation](/docs/basics/prerequisites.md) to make sure you understand how the below commands
work and avoid running into errors.


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
number of GPUs if you want to run them on full data. If you just want to try out the full pipeline, we recommend to subsample
the dataset by e.g. running

```bash
mv raw_aops_data.jsonl raw_aops_data_full.jsonl && head -n 1000 raw_aops_data_full.jsonl > raw_aops_data.jsonl
```

## Problem generation pipeline

### Model conversion

We used Qwen2.5-32B-Instruct to process the AoPS problems, here are the steps to download/convert this model.

```

```