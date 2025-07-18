# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

## Download models

Get the models from HF. E.g.

```bash
huggingface-cli download nvidia/OpenReasoning-Nemotron-1.5B --local-dir OpenReasoning-Nemotron-1.5B
```

To evaluate HLE we used Qwen2.5-32B-Instruct model as a judge. You will need to download it as well if you want
to reproduce HLE numbers

```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir Qwen2.5-32B-Instruct
```

## Prepare evaluation data

```bash
ns prepare_data aai aime24 aime25 hmmt_feb25 brumo25 livecodebench gpqa mmlu-pro hle
```

## Run evaluation

!!! note

    The current script only runs standard evaluation without GenSelect. We will add instructions for GenSelect
    evaluation in the next few days.

We provide an evaluation script in [recipes/openreasoning/eval.py](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openreasoning/eval.py).
It will run evaluation on all benchmarks and for all 4 model sizes. You can modify it directly to change evaluation settings
or to only evaluate a subset of models / benchmarks.