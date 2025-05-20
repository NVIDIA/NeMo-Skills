# NeMo Skills


NeMo-Skills is a collection of pipelines to improve "skills" of large language models. You can use it to generate synthetic data, train/evaluate models, analyzing outputs and more!
Here are some of the things we support.

- [Flexible inference](https://nvidia.github.io/NeMo-Skills/basics/inference): Seamlessly switch between API providers, local server and large-scale slurm jobs for LLM inference.
- [Multiple formats](https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion): Use any of the [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang)
  and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) servers and easily convert checkpoints from one format to another.
- [Model evaluation](https://nvidia.github.io/NeMo-Skills/pipelines/evaluation): Evaluate your models on many popular benchmarks
    - Math problem solving: math, aime24, aime25, omni-math (and many more)
    - Formal proofs in Lean: minif2f, proofnet
    - Coding skills: human-eval, mbpp
    - Chat/instruction following: ifeval, arena-hard, mt-bench
    - General knowledge: mmlu, mmlu-pro, gpqa
- [Model training](https://nvidia.github.io/NeMo-Skills/pipelines/training): Train models at speed-of-light using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/).

You can find the full documentation [here](https://nvidia.github.io/NeMo-Skills/).
To get started, follow this [tutorial](https://nvidia.github.io/NeMo-Skills/basics),
browse available [pipelines](https://nvidia.github.io/NeMo-Skills/pipelines) or run `ns --help` to see all available
commands and their options.

## OpenMathReasoning Dataset

Using our pipelines we created [OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning).
This dataset contains

* 540K unique mathematical problems sourced from [AoPS forums](https://artofproblemsolving.com/community),
* 3.2M long chain-of-thought (CoT) solutions
* 1.7M long tool-integrated reasoning (TIR) solutions
* 566K samples that select the most promising solution out of many candidates (GenSelect)

We used [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) to preprocess problems, and
[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) to generate solutions.

This dataset was a foundation of our winning submission to the
[AIMO-2 Kaggle competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard).

See our [paper](https://arxiv.org/abs/2504.16891) to learn more details!

## OpenMath-Nemotron Models

To demonstrate the quality of this dataset, we release a series of OpenMath-Nemotron models trained on this data.

* [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)
* [OpenMath-Nemotron-7B](https://huggingface.co/nvidia/OpenMath-Nemotron-7B)
* [OpenMath-Nemotron-14B](https://huggingface.co/nvidia/OpenMath-Nemotron-14B)
* [OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle) (this is the model used in [AIMO-2 Kaggle competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard))
* [OpenMath-Nemotron-32B](https://huggingface.co/nvidia/OpenMath-Nemotron-32B)

![Evaluation Results](./docs/openmath-results.png)

The models achieve state-of-the-art results on popular mathematical benchmarks. We present metrics as pass@1 (maj@64) where pass@1
is an average accuracy across 64 generations and maj@64 is the result of majority voting.
Please see our [paper](https://arxiv.org/abs/2504.16891) for more details on the evaluation setup.

| Model                         | AIME24 |  AIME25     |  HMMT-24-25     | HLE-Math    |
|-------------------------------|-----------------|-------|-------|-------------|
| DeepSeek-R1-Distill-Qwen-1.5B | 26.8 (60.0)     | 21.4 (36.7) | 14.2 (26.5) | 2.9 (5.0)   |
| [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) CoT   | 61.6 (80.0)     | 49.5 (66.7) | 39.9 (53.6) | 5.4 (5.4)   |
| [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) TIR   | 52.0 (83.3)     | 39.7 (70.0) | 37.2 (60.7) | 2.5 (6.2)   |
| + Self GenSelect              | 83.3            | 70.0  | 62.2  | 7.9         |
| + 32B GenSelect               | 83.3            | 70.0  | 62.8  | 8.3         |
| DeepSeek-R1-Distill-Qwen-7B  | 54.4 (80.0)     | 38.6 (53.3) | 30.6 (42.9) | 3.3 (5.2)   |
| [OpenMath-Nemotron-7B](https://huggingface.co/nvidia/OpenMath-Nemotron-7B) CoT    | 74.8 (80.0)     | 61.2 (76.7) | 49.7 (57.7) | 6.6 (6.6)   |
| [OpenMath-Nemotron-7B](https://huggingface.co/nvidia/OpenMath-Nemotron-7B) TIR    | 72.9 (83.3)     | 57.5 (76.7) | 54.6 (66.3) | 7.8 (10.8)  |
| + Self GenSelect              | 86.7            | 76.7  | 68.4  | 11.5        |
| + 32B GenSelect               | 86.7            | 76.7  | 69.9  | 11.9        |
| DeepSeek-R1-Distill-Qwen-14B | 65.8 (80.0)     | 48.4 (60.0) | 40.1 (52.0) | 4.2 (4.8)   |
| [OpenMath-Nemotron-14B-MIX (kaggle)](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle) | 73.7 (86.7) | 57.9 (73.3) | 50.5 (64.8) | 5.7 (6.5)   |
| [OpenMath-Nemotron-14B](https://huggingface.co/nvidia/OpenMath-Nemotron-14B) CoT   | 76.3 (83.3)     | 63.0 (76.7) | 52.1 (60.7) | 7.5 (7.6)   |
| [OpenMath-Nemotron-14B](https://huggingface.co/nvidia/OpenMath-Nemotron-14B) TIR   | 76.3 (86.7)     | 61.3 (76.7) | 58.6 (70.9) | 9.5 (11.5)  |
| + Self GenSelect              | 86.7            | 76.7  | 72.4  | 14.1        |
| + 32B GenSelect               | 90.0            | 76.7  | 71.9  | 13.7        |
| QwQ-32B                       | 78.1 (86.7)     | 66.5 (76.7) | 55.9 (63.3) | 9.0 (9.5)   |
| DeepSeek-R1-Distill-Qwen-32B | 66.9 (83.3)     | 51.8 (73.3) | 39.9 (51.0) | 4.8 (6.0)   |
| [OpenMath-Nemotron-32B](https://huggingface.co/nvidia/OpenMath-Nemotron-32B) CoT   | 76.5 (86.7)     | 62.5 (73.3) | 53.0 (59.2) | 8.3 (8.3)   |
| [OpenMath-Nemotron-32B](https://huggingface.co/nvidia/OpenMath-Nemotron-32B) TIR   | 78.4 (93.3)     | 64.2 (76.7) | 59.7 (70.9) | 9.2 (12.5)  |
| + Self GenSelect              | 93.3            | 80.0  | 73.5  | 15.7        |
| DeepSeek-R1                   | 79.1 (86.7)     | 64.3 (73.3) | 53.0 (59.2) | 10.5 (11.4) |

We provide all instructions to [fully reproduce our results](https://nvidia.github.io/NeMo-Skills/openmathreasoning1).

## Nemo Inspector

We recommend this convenient [tool](https://github.com/NVIDIA/NeMo-Inspector) for visualizing inference and data analysis.

## Papers

If you find our work useful, please consider citing us!

```bibtex
@article{moshkov2025aimo2,
  title   = {{AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset}},
  author  = {Ivan Moshkov and Darragh Hanley and Ivan Sorokin and Shubham Toshniwal and Christof Henkel and Benedikt Schifferer and Wei Du and Igor Gitman},
  year    = {2025},
  journal = {arXiv preprint arXiv:2504.16891}
}
```

```bibtex
@inproceedings{toshniwal2024openmathinstruct2,
  title   = {{OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data}},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2025},
  booktitle = {ICLR},
}
```

```bibtex
@inproceedings{toshniwal2024openmathinstruct1,
  title   = {{OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset}},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  booktitle = {NeurIPS},
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.
