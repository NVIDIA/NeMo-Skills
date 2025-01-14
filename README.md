# NeMo Skills

NeMo-Skills is a collection of pipelines to improve "skills" of large language models.
We mainly focus on the ability to solve mathematical problems, but you can use our pipelines for many other tasks as well.
Here are some of the things we support.

- [Flexible inference](https://nvidia.github.io/NeMo-Skills/basics/inference): Seamlessly switch between API providers, local server and large-scale slurm jobs for LLM inference.
- [Multiple formats](https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion): Use any of the [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm)
  and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) servers and easily convert checkpoints from one format to another.
- [Model evaluation](https://nvidia.github.io/NeMo-Skills/pipelines/evaluation): Evaluate your models on many popular benchmarks
    - Math problem solving: gsm8k, math, amc23, aime24, omni-math (and many more)
    - Coding skills: human-eval, mbpp
    - Chat/instruction following: ifeval, arena-hard
    - General knowledge: mmlu (generative)
- [Model training](https://nvidia.github.io/NeMo-Skills/pipelines/training): Train models at speed-of-light using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/).

You can find the full documentation [here](https://nvidia.github.io/NeMo-Skills/).

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

We provide all instructions to [fully reproduce our results](https://nvidia.github.io/NeMo-Skills/openmathinstruct2).

See our [paper](https://arxiv.org/abs/2410.01560) for ablations studies and more details!

## Nemo Inspector

We also provide a convenient [tool](https://github.com/NVIDIA/NeMo-Inspector) for visualizing inference and data analysis.


## Papers

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmathinstruct2,
  title   = {{OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data}},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2410.01560}
}
```

```bibtex
@inproceedings{toshniwal2024openmathinstruct1,
  title   = {{OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset}},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.