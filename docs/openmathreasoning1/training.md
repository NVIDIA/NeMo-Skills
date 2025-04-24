# Model training

We assume you have `/workspace` defined in your [cluster config](../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

## Download data

Get the data from [HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathReasoning).
This might take a while (depending on your network connection) and will use a significant amount of RAM.

```python
from datasets import load_dataset

dataset = load_dataset("nvidia/OpenMathReasoning")

dataset["cot"].to_json("omr-cot.jsonl")
dataset["tir"].to_json("omr-tir.jsonl")
dataset["genselect"].to_json("omr-genselect.jsonl")
```

## Convert to SFT format

Convert the data into the SFT format that NeMo-Aligner understands.

```python
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

cmd = (
    "python -m nemo_skills.training.prepare_data "
        "++prompt_template=qwen-instruct "
        "++preprocessed_dataset_files=/workspace/omr-{inference_mode}.jsonl "
        "++output_key=generated_solution "
        "++output_path=/workspace/omr-{inference_mode}-sft.jsonl "
        "++filters.remove_len_outlier_problems=false "
        "++filters.drop_multi_boxed=false "
        "++filters.trim_prefix=false "
        "++filters.trim_solutions=false "
        "++filters.drop_incorrect_arithmetic=false "
        "++filters.split_arithmetic=false "
        "++filters.remove_contaminated=false "
        "{extra_args} "
)

for inference_mode, extra_args in [("genselect", "")]:
    run_cmd(
        ctx=wrap_arguments(cmd.format(inference_mode=inference_mode, extra_args=extra_args)),
        cluster="local",
    )
```

## Prepare base model

Download the base model and convert it to NeMo format.
The instructions below are for Llama3.1-8B, but the same commands should work for 70B model as well.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir Llama-3.1-8B

ns convert \
    --cluster=local \
    --input_model=/workspace/Llama-3.1-8B \
    --output_model=/workspace/llama3.1-8b-nemo \
    --convert_from=hf \
    --convert_to=nemo \
    --model_type=llama \
    --num_gpus=1 \
    --hf_model_name=meta-llama/Llama-3.1-8B
```

## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `--num_training_jobs=N`.

```bash
ns train \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-8b \
    --output_dir=/workspace/openmathinstruct2-repro/checkpoints \
    --nemo_model=/workspace/llama3.1-8b-nemo \
    --num_nodes=8 \
    --num_gpus=8 \
    --average_steps=10000,20000,30000,40000,50000,60000 \
    --training_data=/workspace/openmathinstruct2-sft.jsonl \
    ++model.data.train_ds.micro_batch_size=8 \
    ++model.tensor_model_parallel_size=4 \
    ++model.pipeline_model_parallel_size=1 \
    ++model.optim.lr=2e-5 \
    ++trainer.sft.save_interval=10000 \
    ++trainer.sft.max_steps=60000 \
    ++trainer.sft.max_epochs=100
```

For 70B model, we used 5M data subset and the following parameters, but training
it longer is likely going to improve results.

```bash
ns train \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-70b \
    --output_dir=/workspace/openmathinstruct2-repro-70b/checkpoints \
    --nemo_model=/workspace/llama3.1-70b-nemo \
    --num_nodes=32 \
    --num_gpus=8 \
    --average_steps=3330,6660,9990,13320,16650,20000 \
    --training_data=/workspace/openmathinstruct2-sft-5M.jsonl \
    ++model.data.train_ds.micro_batch_size=1 \
    ++model.tensor_model_parallel_size=8 \
    ++model.pipeline_model_parallel_size=2 \
    ++model.optim.lr=1e-5 \
    ++trainer.sft.save_interval=3330 \
    ++trainer.sft.max_steps=20000 \
    ++trainer.sft.max_epochs=100
```

If you have a job timeout, it's necessary to set the maximum time per run to 40 minutes
before the timeout to allow for the final checkpoint to be saved. E.g. if your timeout is 4 hours,
add `++exp_manager.max_time_per_run=00:03:20:00`


If you want to follow up with checkpoint conversion and evaluation, see
[training docs](../pipelines/training.md#chaining-pipelines-with-python) for an example of how to do it
through a convenient Python API.
