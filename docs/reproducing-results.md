# How to reproduce our results

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

Please note that unless you have an access to a large GPU cluster, it might take a long time
for some of the commands to complete!

All commands were tested on a cluster with 8 80Gb A100 GPUs per node (or a local machine with the same setup).
If you're using different GPU configuration, change the commands accordingly and
expect ~1% variation in results.

**To ensure exact reproducibility of the results, we recommend checking out the v0.1.1 branch of the repository:**

```bash
git checkout v0.1.1
```

## Evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for Mistral-7B model as an example. They are identical for all models,
except we use batch size of 16 for 34B+ model sizes.

Note that this is not the most efficient configuration for running inference with these models.
We refer you to the [TensoRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
documentation to learn how to make inference more efficient.

1. Get the model from HuggingFace

   ```
   git clone https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf
   ```

2. Convert the model to TensorRT-LLM format for fastest evaluation.

   ```
   docker run --rm --gpus all --ipc=host -v <path to nemo-skills repo>:/code -v <path to OpenMath-Mistral-7B-v0.1-hf>:/model igitman/nemo-skills-trtllm:0.3.0 \
   bash -c ' \
   export PYTHONPATH=/code && cd /code && \
   python nemo_skills/conversion/hf_to_trtllm.py \
      --model_dir /model \
      --output_dir /tmp/trtllm \
      --dtype bfloat16 \
      --tp_size 8 && \
   trtllm-build \
      --checkpoint_dir /tmp/trtllm \
      --output_dir /code/openmath-mistral-7b-trtllm \
      --gpt_attention_plugin bfloat16 \
      --gemm_plugin bfloat16 \
      --context_fmha "enable" \
      --max_input_len 4096 \
      --max_output_len 512 \
      --max_batch_size 64 &&
   cp /model/tokenizer.model /code/openmath-mistral-7b-trtllm'
   ```

3. Run greedy decoding for all datasets. You can increase number of jobs if running on Slurm cluster for faster evaluation.

   ```
   python pipeline/run_eval.py \
     --model_path `pwd`/openmath-mistral-7b-trtllm \
     --server_type tensorrt_llm \
     --output_dir `pwd`/openmath-mistral-7b-eval-results \
     --benchmarks gsm8k:0 asdiv:0 gsm-hard:0 mawps:0 svamp:0 tabmwp:0 algebra222:0 math:0 \
     --num_gpus 8 \
     --num_jobs 1 \
     +prompt=openmathinstruct/sft \
     ++prompt.few_shot_examples.num_few_shots=0 \
     ++split_name=test \
     ++server.code_execution.max_code_executions=6 \
     ++server.code_execution.stop_on_code_error=False \
     ++batch_size=64
   ```

4. Run 50 samples for gsm8k and math to get self-consistency numbers.

   ```
   python pipeline/run_eval.py \
     --model_path `pwd`/openmath-mistral-7b-trtllm \
     --server_type tensorrt_llm \
     --output_dir `pwd`/openmath-mistral-7b-eval-results \
     --benchmarks gsm8k:50 math:50 \
     --num_gpus 8 \
     --num_jobs 1 \
     +prompt=openmathinstruct/sft \
     ++prompt.few_shot_examples.num_few_shots=0 \
     ++skip_filled=True \
     ++split_name=test \
     ++server.code_execution.max_code_executions=6 \
     ++server.code_execution.stop_on_code_error=False \
     ++batch_size=64
   ```

   If the above command fails with mpi error, try to reduce number of samples
   per run, e.g. use `gsm8k:10 math:10` and then do another run with
   `--starting_seed 10` and so on.

5. Compute and summarize all metrics.

   ```
   python pipeline/summarize_results.py `pwd`/openmath-mistral-7b-eval-results
   ```

   If you get permission errors, run the command with sudo or change permissions of the results folder.

You can also download the generation results of all of our released models from [here](https://openmath-test-predictions.s3.amazonaws.com/openmath-test-predictions.zip).

## Dataset generation

To re-create the masked versions of the GSM8K and MATH datasets, you can follow the steps
[here](/docs/synthetic-data-generation.md#masked-solutions).

To re-create the GSM8K version of the [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) dataset
you can run the following:

1. Convert [Mixtral-8x7B base](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model to TensorRT-LLM format.
   Refer to the instructions from steps 1-2 in the [reproducing evaluation](#evaluation) section as well as
   [checkpoint conversion docs](/docs/checkpoint-conversion.md). Use `--max_input_len 4096` and `--max_output_len 512`
   but you might need to change the other parameters based on the exact GPUs configuration.

2. GSM8K portion of the dataset.

   Without reference solutions:

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./synthetic-solutions/gsm8k/ \
     --num_gpus 8 \
     --num_runs 128 \
     +prompt=openmathinstruct/base \
     ++prompt.few_shot_examples.examples_type=gsm8k_text_with_code \
     ++prompt.context_type=empty \
     ++dataset=gsm8k \
     ++split_name=train_full
   ```

   With masked reference solutions:

   ```
   python pipeline/run_labeling.py \
     --model_path <path to trtllm model> \
     --server_type tensorrt_llm \
     --output_dir ./synthetic-solutions/gsm8k-masked/ \
     --num_gpus 8 \
     --num_runs 128 \
     +prompt=openmathinstruct/base \
     ++prompt.few_shot_examples.examples_type=gsm8k_text_with_code \
     ++prompt.context_type=masked_solution \
     ++dataset=gsm8k-masked \
     ++split_name=train_full
   ```

   For the MATH portion of the dataset, change `gsm8k -> math` and use 224 runs.
   Additionally, we generated 32 samples for each of the following example types
   corresponding to the subject-specific prompts we created.

   ```
   math_algebra, math_probability, math_prealgebra, math_number_theory,
   math_geometry, math_precalculus, math_intermediate_algebra
   ```

   Make sure to use different output folder for each generation to not override results.

## Model finetuning

1. Download [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) dataset from HuggingFace, e.g.

   ```
   mkdir open-math-instruct-1
   wget https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/main/correct_solutions/train.jsonl?download=true -O open-math-instruct-1/train.jsonl
   wget https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/main/correct_solutions/validation.jsonl?download=true -O open-math-instruct-1/validation.jsonl
   ```

2. Convert the data to the format that [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/) understands.
   Make sure that NEMO_SKILLS_DATA environment variable is defined
   (see [prerequisites](/docs/prerequisites.md) for more information).

   ```
   python nemo_skills/finetuning/prepare_sft_data.py \
       ++preprocessed_dataset_files="open-math-instruct-1/train.jsonl open-math-instruct-1/validation.jsonl" \
       ++output_path=$NEMO_SKILLS_DATA/sft-data.jsonl \
       ++downsampling_method=fair \
       ++num_output_samples=1024000 \
       ++text_filter_type=any_code \
       ++trim_solutions=True
   ```

3. Get the model you want to finetune, e.g. https://huggingface.co/mistralai/Mistral-7B-v0.1. Convert the model
   to nemo format by running steps from [checkpoint conversion](/docs/checkpoint-conversion.md#huggingface-to-nemo) docs.

4. Run finetuning. The commands below are assumed to be run on a Slurm cluster, but you can modify them to run
   locally for small enough models. Here is an example command for Mistral-7b

   ```
   python pipeline/run_pipeline.py \
      --expname openmath-mistral-7b \
      --nemo_model <path to the nemo model> \
      --stages sft prepare_eval \
      --num_nodes 8 \
      --num_gpus 8 \
      --config-file sft_config_codegen \
      --with_sandbox \
      ++model.data.train_ds.file_path=/data/sft-data.jsonl \
      ++trainer.sft.max_epochs=4 \
      ++trainer.sft.val_check_interval=4000 \
      ++model.tensor_model_parallel_size=4 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.optim.lr=1e-6
   ```

   The finetuned model will be available inside `$NEMO_SKILLS_RESULTS` folder.

   For other models modify the above command according to the following table

   |                    | **Epochs** | **LR** | **# of GPUs** | **TP** | **PP** |
   |--------------------|------------|--------|---------------|--------|--------|
   | **Mistral-7B**     | 4          | 1e-6   | 64            | 4      | 1      |
   | **CodeLlama-7B**   | 4          | 2e-5   | 64            | 4      | 1      |
   | **CodeLlama-13B**  | 4          | 2e-5   | 64            | 4      | 1      |
   | **CodeLlama-34B**  | 4          | 1e-5   | 128           | 8      | 1      |
   | **Llama2-70B**     | 2          | 1e-5   | 256           | 8      | 2      |
   | **CodeLlama-70B**  | 3          | 1e-5   | 256           | 8      | 2      |

Note that the above configuration is not the most efficient way to train these models,
but this is what we used in our project. We refer you to the
[NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/)
documentation to learn how to make training more efficient.