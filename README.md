# NeMo Skills

In this repository we provide a pipeline to improve "skills" of large language models (LLMs). Currently we focus on the ability
to solve simple mathematical problems, but more skills are coming (such as coding and table understanding).

Our pipeline consists of 3 steps and can be directly applied to any LLM that is supported in
[NVIDIA's NeMo Toolkit](https://github.com/NVIDIA/NeMo).

1. <b>[Setup](#supported-models-and-datasets)</b>
   - Pick a "student" model that you want to improve.
     E.g. [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1).
   - [optionally] Pick a "teacher" model (can also use the student model itself).
     E.g. [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1).
   - Choose evaluation benchmarks and training datasets.
     E.g. [GSM8K](https://github.com/openai/grade-school-math) and [MATH](https://github.com/hendrycks/math).
2. <b>[Generate synthetic data](/docs/synthetic-data-generation.md)</b>
   - Write a couple of examples of solutions that you want the student LLM to learn.
     E.g. [teach it to use code](/nemo_skills/inference/prompt/few_shot_examples/examples_gsm8k.py) to solve math problems.
   - Run a large-scale generation of diverse solutions on the training datasets showing your examples in the prompt to the teacher model.
   - Filter the generated solutions based on correctness and quality.
3. <b>[Finetune the student model on the generated dataset](/docs/finetuning.md)</b>

We release a series of [OpenMath models](https://huggingface.co/collections/nvidia/openmath-65c5619de2ba059be0775014)
improved with this pipeline that are one of the best open models for solving mathematical problems and are currently
the only state-of-the-art open models that do not rely on OpenAI for data generation!

<table>
  <tr>
    <td></td>
    <td colspan="2" style="text-align: center;">greedy</td>
    <td colspan="2" style="text-align: center;">majority@50</td>
  </tr>
  <tr>
    <td style="text-align: center;">model</td>
    <td style="text-align: center;">GSM8K</td>
    <td style="text-align: center;">MATH</td>
    <td style="text-align: center;">GMS8K</td>
    <td style="text-align: center;">MATH</td>
  </tr>
  <tr>
    <td style="text-align: right;">GPT-4 <a href="https://arxiv.org/abs/2312.08935">[1]</a></td>
    <td style="text-align: center;">94.4</td>
    <td style="text-align: center;">56.2</td>
    <td style="text-align: center;">-</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td style="text-align: right;">GPT-4 + code <a href="https://arxiv.org/abs/2308.07921v1">[2]</a></td>
    <td style="text-align: center;">92.9</td>
    <td style="text-align: center;">69.7</td>
    <td style="text-align: center;">-</td>
    <td style="text-align: center;">-</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-7B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-7b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-7b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">75.9</td>
    <td style="text-align: center;">43.6</td>
    <td style="text-align: center;">84.8</td>
    <td style="text-align: center;">55.6</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-Mistral-7B (<a href="https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf">HF</a>)</td>
    <td style="text-align: center;">80.2</td>
    <td style="text-align: center;">44.5</td>
    <td style="text-align: center;">86.9</td>
    <td style="text-align: center;">57.2</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-13B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-13b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-13b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">78.8</td>
    <td style="text-align: center;">45.5</td>
    <td style="text-align: center;">86.8</td>
    <td style="text-align: center;">57.6</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-34B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-34b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-34b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">80.7</td>
    <td style="text-align: center;">48.3</td>
    <td style="text-align: center;">88.0</td>
    <td style="text-align: center;">60.2</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-Llama2-70B (<a href="https://huggingface.co/nvidia/OpenMath-Llama-2-70b">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-Llama-2-70b-hf">HF</a>)</td>
    <td style="text-align: center;"><b>84.7</b></td>
    <td style="text-align: center;">46.3</td>
    <td style="text-align: center;">90.1</td>
    <td style="text-align: center;">58.3</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-70B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-70b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-70b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">84.6</td>
    <td style="text-align: center;"><b>50.7</b></td>
    <td style="text-align: center;"><b>90.8</b></td>
    <td style="text-align: center;"><b>60.4</b></td>
  </tr>
</table>


We also release [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1),
a math instruction tuning dataset with 1.8M problem-solution pairs generated using permissively licensed
[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model.

Please see our paper ["OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset"](https://arxiv.org/abs/2402.10176)
for more details!

## Getting started

Try to [run inference with our models](/docs/inference.md) with just a few commands!

We provide all instructions to [fully reproduce our results](/docs/reproducing-results.md).

If you want to improve your own models or to learn more about our pipeline, read through the relevant docs below.

- [Generating synthetic data](/docs/synthetic-data-generation.md)
- [Finetuning models](/docs/finetuning.md)
- [Evaluating models](/docs/evaluation.md)

We also provide a convinient [tool](/nemo_inspector/Readme.md) for visualizing inference and data analysis
Overview |  Inference Page | Analyze Page
:-------------------------:|:-------------------------:|:-------------------------:
[![Demo of the tool](/nemo_inspector/images/demo.png)](https://www.youtube.com/watch?v=EmBFEl7ydqE)   |  [![Demo of the inference page](/nemo_inspector/images/inference_page.png)](https://www.youtube.com/watch?v=6utSkPCdNks) | [![Demo of the analyze page](/nemo_inspector/images/analyze_page.png)](https://www.youtube.com/watch?v=cnPyDlDmQXg)

## Supported models and datasets

Any model that is supported by [NeMo](https://github.com/NVIDIA/NeMo) can be used as a "student".
Many popular models are supported, e.g. [LLaMA2](https://llama.meta.com/llama2/),
[CodeLLaMA](https://llama.meta.com/llama2/),
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and
[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1).
For the "teacher" you can use virtually any openly available LLM, since only inference support is needed.

We currently support the following datasets.

Evaluation:
- [GSM8K](https://github.com/openai/grade-school-math)
- [MATH](https://github.com/hendrycks/math)
- [SVAMP](https://github.com/arkilpatel/SVAMP)
- [GSM-Hard](https://huggingface.co/datasets/reasoning-machines/gsm-hard)
- [ASDiv](https://github.com/chaochun/nlu-asdiv-dataset)
- [ALGEBRA-222](https://github.com/joyheyueya/declarative-math-word-problem)
- [MAWPS](https://github.com/sroy9/mawps)
- [TabMWP](https://github.com/lupantech/PromptPG)

Training:
- [GSM8K](https://github.com/openai/grade-school-math)
- [MATH](https://github.com/hendrycks/math)

Please check out [evaluation](/docs/evaluation.md) and [finetuning](/docs/finetuning.md) sections to learn more!

## Paper and Citation

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmath,
  title   = {OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2402.10176}
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.