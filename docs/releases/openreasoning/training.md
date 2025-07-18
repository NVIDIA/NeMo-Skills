# Model training

We mostly use the same training commands as for [OpenMathReasoning models](../openmathreasoning/training.md#run-training). The only difference
is that we pack sequences to 49152 length and use a little different hyperparameters detailed in the following table.

|                       | **lr** | **min_lr** | **TP** | **PP** | **CP** |
| --------------------- | ------ | ---------- | ------ | ------ | ------ |
| **Qwen2.5-Math-1.5B** | 1e-4   | 1e-7       | 1      | 1      | 4      |
| **Qwen2.5-Math-7B**   | 1e-4   | 1e-7       | 4      | 1      | 4      |
| **Qwen2.5-14B**       | 1e-4   | 1e-7       | 8      | 1      | 4      |
| **Qwen2.5-32B**       | 1e-4   | 1e-7       | 8      | 2      | 4      |

All models are trained for 30000 steps with a single round of SFT and we take the last checkpoint as the final model.