# LLM-based data decontamination

!!! info

    This pipeline starting script is [nemo_skills/pipeline/check_contamination.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/check_contamination.py)

    All extra parameters are passed to [nemo_skills/inference/check_contamination.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/inference/check_contamination.py)

We implemented an LLM-based data decontamination pipeline following
[lmsys methodology](https://lmsys.org/blog/2023-11-14-llm-decontaminator/).

There are two main ways how you can use this pipeline: to check existing dataset
for contamination and to decontaminate the training dataset by removing all
contaminated questions.

## To check for contamination

Let's say you want to check for contamination of [MATH](https://github.com/hendrycks/math)
training set with MATH, AMC-23 and AIME-24 test sets. First, get the data

```bash
python -m nemo_skills.dataset.prepare math amc23 aime24
```

Then we need to retrieve top-k similar questions from the training set. Assuming
you have `/workspace` defined in your [cluster config](../basics/prerequisites.md#cluster-configs)
you can do it in the following way

```python
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import run_cmd


test_sets = ['math', 'amc23', 'aime24']
retrieve_from = ",".join(f"/nemo_run/code/nemo_skills/dataset/{test_set}/test.jsonl" for test_set in test_sets)

cmd = (
    f"python -m nemo_skills.inference.retrieve_similar "
    f"    ++retrieve_from=\\\'{retrieve_from}\\\' "
    f"    ++compare_to='/nemo_run/code/nemo_skills/dataset/math/train_full.jsonl' "
    f"    ++output_file='/workspace/math-contamination-retrieved.jsonl' "
    f"    ++top_k=1 "
)

run_cmd(
    cluster="local",
    container="nemo",
    num_gpus=1,  # can increase this if you have more gpus
    ctx=wrap_arguments(cmd),
)
```

Next, you need to run LLM inference to check those closest found questions from the output file. Here is an example
using Llama-405B from Nvidia API catalog, but you can replace it with OpenAI models or self-hosted models.

```
ns check_contamination \
    --cluster=local \
    --input_file=/workspace/math-contamination-retrieved.jsonl \
    --output_file=/workspace/math-contamination-results.jsonl \
    --server_type=openai \
    --model=meta/llama-3.1-405b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1
```

This script will print an output that looks like this

```
Contamination portion: 13.91% (705/5070)
```

## To decontaminate training data

If you want instead to clean your training data from contaminated examples all the commands stay the same, but
you need to swap values for the `retrieve_from` and `compare_to` arguments in the `retrieve_similar` step
since we now want to make a check for each training set example and find closest test set problems.

After you get `/workspace/math-contamination-results.jsonl`, you can pass it into [prepare_data command](training.md#preparing-the-data)
with `++contamination_file=...` option.

See a more detailed example in [OpenMathInstruct-2 dataset construction pipeline](../openmathinstruct2/dataset.md#decontamination).