# OpenMathInstruct-2

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

## Paper

[OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data](https://arxiv.org/abs/2410.01560)

If you find our work useful, please consider citing us!

```bibtex
@inproceedings{toshniwal2024openmathinstruct2,
  title   = {{OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data}},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2025},
  booktitle = {ICLR},
}
```

## How to reproduce our results

Browse the sections below to see all commands needed to fully reproduce our results.

Please note that unless you have an access to a large GPU cluster, it might take a very long time
for some of the commands to complete!

- [Model evaluation](evaluation.md)
- [Dataset construction](dataset.md)
- [Model training](training.md)
