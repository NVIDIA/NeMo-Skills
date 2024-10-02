# LLM-based data decontamination

Make sure to complete [prerequisites](/docs/prerequisites.md).

Please refer to the following docs if you have questions about:
- [Prompt format](/docs/prompt-format.md)
- [Generation parameters](/docs/common-parameters.md)
- [How to self-host models](/docs/generation.md)

We implemented an LLM-based data decontamination pipeline following
[lmsys methodology](https://lmsys.org/blog/2023-11-14-llm-decontaminator/).

There are two main ways how you can use this pipeline - to check existing dataset
for contamination and to decontaminate the training dataset by removing all
contaminated questions.

## To check for contamination

Let's say you want to check for contamination of [MATH](https://github.com/hendrycks/math)
training set with MATH, AMC-23 and AIME-24 test sets.

First, we need to retrieve top-k similar questions from the training set. Assuming
you're running from locally installed repository you can do it in the following way

```
python -m nemo_skills.inference.retrieve_similar \
    ++retrieve_from=./nemo_skills/dataset/math/train_full.jsonl \
    ++compare_to="./nemo_skills/dataset/math/test.jsonl ./nemo_skills/dataset/amc23/test.jsonl ./nemo_skills/dataset/aime24/test.jsonl" \
    ++output_file=./math-contamination-retrieved.jsonl \
    ++top_k=1
```

> **_NOTE:_** Currently the above command doesn't run inside docker, so you will need to install additional packages.
> We will fix it soon by providing the same "pipeline" interface.

Next, you need to run LLM inference to check those closest found questions from the output file. Here is an example
using Llama-405B from Nvidia API catalog, but you can replace it with OpenAI models or self-hosted models.

```
ns check_contamination \
    --cluster local \
    --input_file /workspace/NeMo-Skills/math-contamination-retrieved.jsonl \
    --output_file /workspace/NeMo-Skills/math-contamination-results.jsonl \
    --server_type=openai \
    --model=meta/llama-3.1-405b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1
```

assuming you have a parent dir mounted as `/workspace` in your cluster config. This script will print an output that
looks like this

```
Contamination portion: 13.91% (705/5070)
```

## To decontamination the training data

TBD