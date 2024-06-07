# Synthetic data generation (labeling)

The instructions to synthetically generate new solutions are almost identical to the
[evaluation](/docs/evaluation.md) instructions, since both workflows use the same
scripts, just with different parameters.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

Here are the basic commands to generate 128 solutions for each problem in GSM8K dataset using
any "teacher" model, e.g. [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1).

1. Get the model and follow [instructions](/docs/checkpoint-conversion.md#huggingface-to-tensorrt-llm)
   to convert it to TensorRT-LLM format. While you can do inference with NeMo, we highly
   recommend using TensorRT-LLM for synthetic data generation as it can be up to 10x faster.

2. Start data generation. Note that if you're running locally, all jobs will run sequentially.

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./synthetic-solutions/ \
     --num_gpus <number of GPUs on your machine/cluster node> \
     --num_runs 128 \
     +prompt=openmathinstruct/base \
     ++prompt.few_shot_examples.examples_type=gsm8k_text_with_code \
     ++prompt.context_type=empty \
     ++dataset=gsm8k \
     ++split_name=train_full
   ```

   This will run 128 slurm jobs each generating a solutions with unique random seed. You can customize solution
   format with `++prompt.few_shot_examples.examples_type` (see [nemo_skills/inference/prompt/few_shot_examples](/nemo_skills/inference/prompt/few_shot_examples)) and whether to show reference solution with `++prompt.context_type=reference_solution`. We found
   that showing original solution is generally harmful, so it's recommended to either set `++prompt.context_type=empty` (no
   reference solution in prompt) or to show *masked* reference solution and select `++dataset=gsm8k_masked` and `++prompt.context_type=masked_solution` to use our
   masked version of solutions (see the [paper](https://arxiv.org/abs/2402.10176) for details).

3. You would typically follow by [converting the data to SFT format and finetuning models](/docs/finetuning.md).

For more details read [evaluation](/docs/evaluation.md) docs.

## Masked solutions

We provide masked datasets [GSM8K-Masked](https://huggingface.co/datasets/nvidia/OpenMath-GSM8K-masked) and
[MATH-Masked](https://huggingface.co/datasets/nvidia/OpenMath-MATH-masked) that were generated using Mixtral-8x7b.
Here are the steps to create masked solutions for the different dataset or using other model.

1. Get the model and follow [instructions](/docs/checkpoint-conversion.md#huggingface-to-tensorrt-llm)
   to convert it to TensorRT-LLM format. While you can do inference with NeMo, we highly
   recommend using TensorRT-LLM for synthetic data generation as it can be up to 10x faster.

2. For GSM8K and MATH you can use `++prompt.few_shot_examples.examples_type=gsm8k_generate_masked` and `++prompt.few_shot_examples.examples_type=math_generate_masked` respectively.
   If using other dataset, create few-shot examples that show how to "translate" original reference solution to a masked one.

3. Start data generation. Note that if you're running locally, all jobs will run sequentially.

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./masked-solutions/ \
     --num_gpus <number of GPUs on your machine/cluster node> \
     --num_runs 32 \
     +prompt=openmathinstruct/text_masked_base \
     ++prompt.few_shot_examples.examples_type=gsm8k_generate_masked \
     ++prompt.context_type=reference_solution \
     ++dataset=gsm8k \
     ++split_name=train_full
   ```

This will run 32 slurm jobs with unique random seed, each generating a masked solutions based on reference solutions
and provided few-shot examples.

4. Pick the best masked solutions and convert to the expected format.

   ```
   python nemo_skills/finetuning/prepare_masked_data.py \
     ++dataset=<dataset_name from datasets folder> \
     ++masked_soln_jsonl_files=./masked-solutions/output-rs*.jsonl \
     ++split_name=train_full
   ```

Prepared dataset will be saved in `datasets/<dataset_name>-masked/<split_name>.jsonl`.

Now you can go back to step 2 of the previous section and specify `++dataset=<dataset_name>-masked` and
`++prompt.context_type=masked_solution`.