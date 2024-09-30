# Training an LLM

Make sure to complete [prerequisites](/docs/prerequisites.md).

Please refer to the following docs if you have questions about:
- [Prompt format](/docs/prompt-format.md)
- [Checkpoint conversion](/docs/checkpoint-conversion.md)
- [Evaluation](/docs/evaluation.md)

## Preparing the data

Before running the training we need to prepare the data in the right format. Here is an example command

```
python -m nemo_skills.training.prepare_sft_data \
    ++input_files="<path to the generated synthetic data>/output-rs*.jsonl <you can have multiple paths as well>" \
    ++output_path=sft-data.jsonl \
    ++prompt_config=generic/math \
    ++prompt_template=llama3-instruct \
    ++generation_suffix='\"<|eot_id|>\"'
```

Note that unlike most other scripts, this one doesn't accept a `--cluster` parameter and you can currently only run
it locally (this will be changed soon).

You need to pass in the config/template files so that we can format the data accordingly. There are many more parameters
that data preparation script supports which you can see [here](/nemo_skills/training/data_preparation_utils/prepare_sft_data.yaml).
We are using [SDP library](https://github.com/NVIDIA/NeMo-speech-data-processor) for preparing the data, so it's
a good idea to check their documentation to understand how this config is structured.

> **_NOTE:_** Even though we support both SFT and DPO training, the data preparation is currently only implemented
> for SFT jobs. For DPO, you'd need to manually prepare the data according to the
> [NeMo-Aligner documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html#dpo-model-training).
> We will add a proper support for DPO data preparation in the near future.


## Running training

We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/) to run LLM training,
so you can check their documentation to learn about all supported parameters.

Here is an example of how to run a training job.

ns train \
    --cluster=slurm \
    --expname=my-training-job \
    --output_dir=/workspace/my-training-job/checkpoints \
    --nemo_model=/nemo_models/llama3.1-8b-base \
    --num_nodes=8 \
    --num_gpus=8 \
    --num_training_jobs=4 \
    --training_data=/data/sft-data.jsonl

This will run training on 8 nodes of 8 GPUs, using 4 dependent slurm jobs.
By default we are training for 2 epochs, saving checkpoints every 1000 steps,
but you can adjust these values. It's also recommended to tune micro batch size
and tensor parallel parameters for optimal performance. E.g. these are good
defaults for an 8B model size

```
    ++model.data.train_ds.micro_batch_size=4 \
    ++model.tensor_model_parallel_size=4
```

You can customize any of the SFT parameters by directly providing them, e.g.
to disable wandb logging and add dropout use

```
   --disable_wandb \
   ++model.ffn_dropout=0.1 \
   ++model.attention_dropout=0.1 \
   ++model.hidden_dropout=0.1
```

The training script will average all of your generated checkpoints upon completion
(we found this to consistently increase the downstream accuracy). If you want to
only average a subset of checkpoint, add `--average_steps` parameter (e.g. if you
want to disable averaging, set it to the last training step). If you only want
to average the checkpoints of the finished job, set `--num_training_jobs=0`.

Typically after training we want to follow up with evaluation. You can schedule
an evaluation job right away by providing a `--run_after=my-training-job` argument
which will appropriately set slurm dependencies.

```
ns eval \
    --cluster=slurm \
    --model=/workspace/my-training-job/checkpoints/model-averaged-nemo \
    --server_type=nemo \
    --output_dir=/workspace/my-training-job/results/ \
    --benchmarks gsm8k:0,math:0 \
    --server_gpus=8 \
    --run_after=my-training-job \
    ++prompt_template=llama3-instruct \
    ++batch_size=128
```

In general we don't recommend to run inference using NeMo checkpoints as it is
much slower than other server formats. Here is how you can chain the commands
to schedule checkpoint conversion and evaluation after training
(whenever you need to run multiple commands, it's more convenient to use python interface)

```python
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import train, convert, eval

expname = "my-training-job"
cluster = "slurm"
output_dir = f"/workspace/{expname}/checkpoints"

train(
    ctx=wrap_arguments(""),
    clustercluster,
    expname=expname,
    output_dir=output_dir,
    nemo_model="/nemo_models/llama3.1-8b-base",
    num_nodes=8,
    num_gpus=8,
    num_training_jobs=4,
    training_data="/data/sft-data.jsonl",
)

convert(
    ctx=wrap_arguments(""),
    cluster=cluster,
    input_model=f"{output_dir}/model-averaged-nemo",
    output_model=f"{output_dir}/model-averaged-hf",
    expname=f"{expname}-to-hf",
    run_after=expname,
    convert_from="nemo",
    convert_to="hf",
    num_gpus=8,
    hf_model_name="meta-llama/Meta-Llama-3.1-8B",
)

convert(
    ctx=wrap_arguments(""),
    cluster=cluster,
    input_model=f"{output_dir}/model-averaged-hf",
    output_model=f"{output_dir}/model-averaged-trtllm",
    expname=f"{expname}-to-trtllm",
    run_after=f"{expname}-to-hf",
    convert_from="hf",
    convert_to="trtllm",
    num_gpus=8,
)

eval(
    ctx=wrap_arguments("++prompt_template=llama3-instruct ++batch_size=128"),
    cluster=cluster,
    model=f"{output_dir}/model-averaged-trtllm",
    server_type="trtllm",
    output_dir=f"{output_dir}/results/",
    benchmarks="gsm8k:0,math:0",
    server_gpus=8,
    run_after=f"{expname}-to-trtllm",
)
```
