---
hide:
  - navigation
  - toc
---

[NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) is a collection of pipelines to improve "skills" of large language models (LLMs). We support everything needed for LLM development, from synthetic data generation, to model training, to evaluation on a wide range of benchmarks. Start developing on a local workstation and move to a large-scale Slurm cluster with just a one-line change.

Here are some of the features we support:

- [Flexible LLM inference](basics/inference.md):
    - Seamlessly switch between API providers, local server and large-scale Slurm jobs for LLM inference.
    - Host models (on 1 or many nodes) with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang) or [Megatron](https://github.com/NVIDIA/Megatron-LM).
    - Scale SDG jobs from 1 GPU on a local machine all the way to tens of thousands of GPUs on a Slurm cluster.
- [Model evaluation](pipelines/evaluation.md):
    - Evaluate your models on many popular benchmarks.
        - Math problem solving: hmmt_feb25, brumo25, aime24, aime25, omni-math (and many more)
        - Formal proofs in Lean: minif2f, proofnet
        - Coding skills: swe-bench, scicode, livecodebench, human-eval, mbpp
        - Chat/instruction following: ifbench, ifeval, arena-hard
        - General knowledge: mmlu, mmlu-pro, gpqa
        - Long context: ruler
    - Easily parallelize each evaluation across many Slurm jobs, self-host LLM judges, bring your own prompts or change benchmark configuration in any other way.
- [Model training](pipelines/training.md): Train models using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/), [NeMo-RL](https://github.com/NVIDIA/NeMo-RL/) or [verl](https://github.com/volcengine/verl).


To get started, follow these [steps](basics/index.md), browse available [pipelines](./pipelines/index.md) or run `ns --help` to see all available
commands and their options.

You can find more examples of how to use NeMo-Skills in the [tutorials](./tutorials/index.md) page.

We've built and released many popular models and datasets using NeMo-Skills. See all of them in the [Papers & Releases](./releases/index.md) documentation.