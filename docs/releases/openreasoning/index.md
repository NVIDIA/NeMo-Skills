---
date: 2025-07-18
---

# OpenReasoning

We released OpenReasoning-Nemotrons: a suite of reasoning-capable large language models (LLMs) which have been distilled from the DeepSeek R1 0528 671B model. Trained on a massive, high-quality dataset distilled from the new DeepSeek R1 0528, our new 7B, 14B, and 32B models achieve state-of-the-art performance on a wide range of reasoning benchmarks for their respective sizes in the domain of mathematics, science and code.
The models are available to download from **Hugging Face** ([1.5B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B), [7B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B), [14B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-14B), [32B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B)).

The foundation of these models is their dataset. We generated **5 million high-quality reasoning-based solutions** by leveraging the powerful DeepSeek R1 0528 model across the domains of mathematics, coding, and science. This dataset will be released in the coming months, enabling all models to improve their reasoning capabilities on these domains.

## Evaluation results

![Evaluation Results with pass@1](./pass-1.png)

Our models demonstrate exceptional performance across a suite of challenging reasoning benchmarks. The 7B, 14B, and 32B models consistently set new state-of-the-art records for their size classes.

| **Model** | **AritificalAnalysisIndex*** | **GPQA** | **MMLU-PRO** | **HLE** | **LiveCodeBench*** | **SciCode** | **AIME24** | **AIME25** | **HMMT FEB 25**  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.5B**| 31.0 | 31.6 | 47.5 | 5.5 | 28.6 | 2.2 | 55.5 | 45.6 | 31.5 |
| **7B** | 54.7 | 61.1 | 71.9 | 8.3 | 63.3 | 16.2 | 84.7 | 78.2 | 63.5 |
| **14B** | 60.9 | 71.6 | 77.5 | 10.1 | 67.8 | 23.5 | 87.8 | 82.0 | 71.2 |
| **32B** | 64.3 | 73.1 | 80.0 | 11.9 | 70.2 | 28.5 | 89.2 | 84.0 | 73.8 |

\* This is our estimation of the Artificial Analysis Intelligence Index, not an official score.

\* LiveCodeBench version 6, date range 2408-2505.


## Combining the work of multiple agents
OpenReasoning-Nemotron models can be used in a "heavy" mode by starting multiple parallel generations and combining them together via [generative solution selection (GenSelect)](https://arxiv.org/abs/2504.16891). To add this "skill" we follow the original GenSelect training pipeline except we do not train on the selection summary but use the full reasoning trace of DeepSeek R1 0528 671B instead. We only train models to select the best solution for math problems but surprisingly find that this capability directly generalizes to code and science questions! With this "heavy" GenSelect inference mode, OpenReasoning-Nemotron-32B model surpasses O3 (High) on math and coding benchmarks.

![Evaluation Results with GenSelect](./genselect.png)

| **Model** | **Pass@1 (Avg@64)** | **Majority@64** | **GenSelect** |
| :--- | :--- | :--- | :--- |
| **1.5B** | | | |
| **AIME24** | 55.5 | 76.7 | 76.7 |
| **AIME25** | 45.6 | 70.0 | 70.0 |
| **HMMT Feb 25** | 31.5 | 46.7 | 53.3 |
| **7B** | | | |
| **AIME24** | 84.7 | 93.3 | 93.3 |
| **AIME25** | 78.2 | 86.7 | 93.3 |
| **HMMT Feb 25** | 63.5 | 83.3 | 90.0 |
| **LCB v6 2408-2505** | 63.4 | n/a | 67.7 |
| **14B** | | | |
| **AIME24** | 87.8 | 93.3 | 93.3 |
| **AIME25** | 82.0 | 90.0 | 90.0 |
| **HMMT Feb 25** | 71.2 | 86.7 | 93.3 |
| **LCB v6 2408-2505** | 67.9 | n/a | 69.1 |
| **32B** | | | |
| **AIME24** | 89.2 | 93.3 | 93.3 |
| **AIME25** | 84.0 | 90.0 | 93.3 |
| **HMMT Feb 25** | 73.8 | 86.7 | 96.7 |
| **LCB v6 2408-2505** | 70.2 | n/a | 75.3 |
| **HLE** | 11.8 | 13.4 | 15.5 |



## How to reproduce our results

Browse the sections below to see all commands needed to fully reproduce our results.

Please note that unless you have an access to a large GPU cluster, it might take a very long time
for some of the commands to complete!

- [Model evaluation](evaluation.md)
- [Dataset construction](dataset.md)
- [Model training](training.md)
