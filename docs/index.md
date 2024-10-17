---
hide:
  - navigation
  - toc
---

[NeMo-Skills](https://github.com/Kipok/NeMo-Skills) is a collection of pipelines to improve "skills" of large language models.
We mainly focus on the ability to solve mathematical problems, but you can use our pipelines for many other tasks as well.
Here are some of the things we support.

- [Flexible inference](basics/inference.md): Seamlessly switch between API providers, local server and large-scale slurm jobs for LLM inference.
- [Multiple formats](pipelines/checkpoint-conversion.md): Use any of the [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm)
  and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) servers and easily convert checkpoints from one format to another.
- [Model evaluation](pipelines/evaluation.md): Evaluate your models on many popular benchmarks
    - Math problem solving: gsm8k, math, amc23, aime24, omni-math (and many more)
    - Coding skills: human-eval, mbpp
    - Chat/instruction following: ifeval, arena-hard
    - General knowledge: mmlu (generative)
- [Model training](pipelines/training.md): Train models at speed-of-light using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/).

To get started, follow the [prerequisites](basics/prerequisites.md) and then run `ns --help` to see all available
commands and their options.