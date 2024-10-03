# NeMo Skills

In this repository we provide pipelines to improve "skills" of large language models (LLMs).
Currently we focus on the ability to solve mathematical problems, but you can use our pipelines for many other tasks as well.

Here are some of the things we support.

- Easily [convert](/docs/checkpoint-conversion.md) models between [NeMo](https://github.com/NVIDIA/NeMo),
  [vLLM](https://github.com/vllm-project/vllm) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) formats.
- Host the server in any of the above formats to run [large-scale synthetic data generation](/docs/generation.md).
  You can also call Nvidia NIM API or OpenAI API with the same interface, so it's easy to switch from quick prototyping
  to large-scale slurm jobs.
- [Evaluate your models](/docs/evaluation.md) on many popular benchmarks (it's easy to add new benchmarks or customize
  existing settings). The following benchmarks are supported out-of-the-box
    - Math problem solving: gsm8k, math, amc23, aime24 (and many more)
    - Coding skills: human-eval, mbpp
    - Chat/instruction following: ifeval, arena-hard
    - General knowledge: mmlu (generative)
- [Train models](/docs/training.md) using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/).
- We support other pipelines as well, such as [LLM-based dataset decontamination](/docs/decontamination.md)
  or using [LLM-as-a-judge](/docs/llm-as-a-judge.md). And it's easy to add new workflows!

To get started, follow the [prerequisites](/docs/prerequisites.md) and then run `ns --help` to see all available
commands and their options.

## OpenMathInstruct-2

Using our pipelines we created [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
which consists of 14M question-solution pairs (600K unique questions), making it nearly eight times larger
than the previous largest open-source math reasoning dataset.

The models trained on this dataset achieve strong results on common mathematical benchmarks.

<table>
  <tr>
    <td style="text-align: center;">model</td>
    <td style="text-align: center;">GSM8K</td>
    <td style="text-align: center;">MATH</td>
    <td style="text-align: center;">AMC 2023</td>
    <td style="text-align: center;">AIME 2024</td>
    <td style="text-align: center;">Omni-MATH</td>
  </tr>
  <tr>
    <td style="text-align: right;">Llama3.1-8B-Instruct</td>
    <td style="text-align: center;">84.5</td>
    <td style="text-align: center;">51.9</td>
    <td style="text-align: center;">9/40</td>
    <td style="text-align: center;">2/30</td>
    <td style="text-align: center;">12.7</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath2-Llama3.1-8B (<a href="https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B-nemo">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B">HF</a>)</td>
    <td style="text-align: center;">91.7</td>
    <td style="text-align: center;">67.8</td>
    <td style="text-align: center;">16/40</td>
    <td style="text-align: center;">3/30</td>
    <td style="text-align: center;">22.0</td>
  </tr>
  <tr>
    <td style="text-align: right;">+ majority@256</td>
    <td style="text-align: center;">94.1</td>
    <td style="text-align: center;">76.1</td>
    <td style="text-align: center;">23/40</td>
    <td style="text-align: center;">3/30</td>
    <td style="text-align: center;">24.6</td>
  </tr>
  <tr>
    <td style="text-align: right;">Llama3.1-70B-Instruct</td>
    <td style="text-align: center;">95.1</td>
    <td style="text-align: center;">68.0</td>
    <td style="text-align: center;">19/40</td>
    <td style="text-align: center;">6/30</td>
    <td style="text-align: center;">19.0</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath2-Llama3.1-70B (<a href="https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B-nemo">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B">HF</a>)</td>
    <td style="text-align: center;">94.9</td>
    <td style="text-align: center;">71.9</td>
    <td style="text-align: center;">20/40</td>
    <td style="text-align: center;">4/30</td>
    <td style="text-align: center;">23.1</td>
  </tr>
  <tr>
    <td style="text-align: right;">+ majority@256</td>
    <td style="text-align: center;">96.0</td>
    <td style="text-align: center;">79.6</td>
    <td style="text-align: center;">24/40</td>
    <td style="text-align: center;">6/30</td>
    <td style="text-align: center;">27.6</td>
  </tr>
</table>

We provide all instructions to [fully reproduce our results](/docs/reproducing-results.md).

See our [paper](https://arxiv.org/abs/2410.01560) for ablations studies and more details!

## Nemo Inspector

We also provide a convenient [tool](/nemo_inspector/Readme.md) for visualizing inference and data analysis
|                                              Overview                                               |                                                     Inference Page                                                      |                                                    Analyze Page                                                     |
| :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
| [![Demo of the tool](/nemo_inspector/images/demo.png)](https://www.youtube.com/watch?v=EmBFEl7ydqE) | [![Demo of the inference page](/nemo_inspector/images/inference_page.png)](https://www.youtube.com/watch?v=6utSkPCdNks) | [![Demo of the analyze page](/nemo_inspector/images/analyze_page.png)](https://www.youtube.com/watch?v=cnPyDlDmQXg) |


## Papers

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmath,
  title   = {OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2410.01560}
}
```

```bibtex
@article{toshniwal2024openmath,
  title   = {OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2402.10176}
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.