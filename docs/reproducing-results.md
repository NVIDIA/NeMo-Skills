# How to reproduce our results

Make sure to complete [prerequisites](/docs/prerequisites.md).

If you want to reproduce results for [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)
please check out [v0.1.1](https://github.com/Kipok/NeMo-Skills/blob/v0.1.1/docs/reproducing-results.md)
branch of the repository and read the instructions in there.

```bash
git checkout v0.1.1
```

Below are the instructions for reproducing
[OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data](https://arxiv.org/abs/2410.01560).

Please note that unless you have an access to a large GPU cluster, it might take a very long time
for some of the commands to complete!

## Evaluation

Here are the commands you can run to reproduce our evaluation numbers.
The commands below are for OpenMath-2-Llama3.1-8b model as an example.
We assume you have `/workspace` defined in your cluster config and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

1. Get the model from HF. E.g.

   ```
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download nvidia/OpenMath2-Llama3.1-8B --local-dir OpenMath2-Llama3.1-8B
   ```

2. Convert the model to TensorRT-LLM format. This is optional, but highly recommended for more exact
   results and faster inference. If you skip it, replace `--server_type trtllm` with `--server-type vllm`
   in the commands below and change model path to `/workspace/OpenMath2-Llama3.1-8B`. You might also need
   to set smaller batch size for vllm.

   ```
   ns convert \
       --cluster=local \
       --input_model=/workspace/OpenMath2-Llama3.1-8B \
       --output_model=/workspace/openmath2-llama3.1-8b-trtllm \
       --convert_from=hf \
       --convert_to=trtllm \
       --num_gpus=1 \
       --hf_model_name=nvidia/OpenMath2-Llama3.1-8B
   ```

   Change the number of GPUs if you have more than 1 (required for 70B model).

3. Prepare the data.

   ```
   python -m nemo_skills.dataset.prepare gsm8k math amc23 aime24 omni-math
   ```

4. Run greedy decoding.

   ```
   ns eval \
       --cluster=local \
       --model=/workspace/openmath2-llama3.1-8b-trtllm \
       --server_type=trtllm \
       --output_dir=/workspace/openmath2-llama3.1-8b-eval \
       --benchmarks=aime24:0,amc23:0,math:0,gsm8k:0,omni-math:0 \
       --server_gpus=1 \
       --num_jobs=1 \
       ++prompt_template=llama3-instruct \
       ++batch_size=512 \
       ++inference.tokens_to_generate=4096
   ```

   If running on slurm, you can set `--num_jobs` to a bigger number of -1 to run
   each benchmark in a separate node. The number of GPUs need to match what you used
   in the conversion command.

   After the generation is done, we want to run LLM-as-a-judge evaluation to get more
   accurate numbers than symbolic comparison. You need to define `OPENAI_API_KEY` for
   the command below to work.

   ```
   ns llm_math_judge \
       --cluster=local \
       --model=gpt-4o \
       --server_type=openai \
       --server_address=https://api.openai.com/v1 \
       --input_files="/workspace/openmath2-llama3.1-8b-eval/eval-results/**/output*.jsonl"
   ```

   Finally, to print the metrics run

   ```
   ns summarize_results /workspace/openmath2-llama3.1-8b-eval/eval-results --cluster local
   ```

   This should print the metrics including both symbolic and judge evaluation. The judge is typically more accurate.

   ```
   ------------------------------------------------- aime24 ------------------------------------------------
   evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
   greedy          | 30          | 10.00            | 10.00         | 10.00        | 10.00       | 6.67


   ------------------------------------------------- gsm8k -------------------------------------------------
   evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
   greedy          | 1319        | 90.75            | 91.70         | 90.75        | 91.70       | 0.00


   ----------------------------------------------- omni-math -----------------------------------------------
   evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
   greedy          | 4428        | 18.97            | 22.22         | 18.11        | 23.08       | 2.55


   -------------------------------------------------- math -------------------------------------------------
   evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
   greedy          | 5000        | 67.70            | 68.10         | 67.50        | 68.30       | 1.36


   ------------------------------------------------- amc23 -------------------------------------------------
   evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
   greedy          | 40          | 32.50            | 40.00         | 32.50        | 40.00       | 0.00
   ```

   The numbers may vary by 1-2% depending on the server type, number of GPUs and batch size used.

5. Run majority voting.

   ```
   ns eval \
       --cluster=local \
       --model=/workspace/openmath2-llama3.1-8b-trtllm \
       --server_type=trtllm \
       --output_dir=/workspace/openmath2-llama3.1-8b-eval \
       --benchmarks=aime24:256,amc23:256,math:256,gsm8k:256,omni-math:256 \
       --server_gpus=1 \
       --num_jobs=1 \
       --skip_greedy \
       ++prompt_template=llama3-instruct \
       ++batch_size=512 \
       ++inference.tokens_to_generate=4096
   ```

   This will take a very long time unless you run on slurm cluster. After the generation is done, you will be able
   to see symbolic scores right away. You can evaluate with the judge by first creating new files with majority
   answers. E.g. for "math" benchmark run

   ```
   python -m nemo_skills.evaluation.fill_majority_answer \
       ++input_files="./openmath2-llama3.1-8b-eval/eval-results/math/output-rs*.jsonl" \
       ++fill_key=predicted_answer
   ```

   This will replace `predicted_answer` in all files with majority answer.

   After that, let's copy just a single of those files into a new folder so that we can run the llm-judge pipeline
   on them.

   ```
   mkdir -p ./openmath2-llama3.1-8b-eval/eval-results-majority/math
   cp ./openmath2-llama3.1-8b-eval/eval-results/math/output-rs0.jsonl ./openmath2-llama3.1-8b-eval/eval-results-majority/math/
   ```

   Repeat the above steps for all benchmarks. Now we are ready to run the judge pipeline and summarize results
   after it is finished. You need to define `OPENAI_API_KEY` for the command below to work.

   ```
   ns llm_math_judge \
       --cluster=local \
       --model=gpt-4o \
       --server_type=openai \
       --server_address=https://api.openai.com/v1 \
       --input_files="/workspace/openmath2-llama3.1-8b-eval/eval-results-majority/**/output*.jsonl"
   ```

   ```
   ns summarize_results /workspace/openmath2-llama3.1-8b-eval/eval-results-majority --cluster local
   ```

   This will print majority results (they will be labeled as `majority@1` since we fused them into a single file).
   You can also ignore the symbolic score as it's not accurate anymore after we filled majority answers.

## Dataset construction

Here are the commands you can run to re-create [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
We assume you have `/workspace` defined in your cluster config and are running
all commands on a slurm cluster. Change the commands accordingly if running locally
(but it's going to take a lot of time).
We also assume you have the [Llama3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
on that cluster inside `/trt_models/llama-3.1-405b-instruct` (should be mounted in your config)
that's been [converted](/docs/checkpoint-conversion.md) to TensorRT-LLM format.
See [generation docs](/docs/generation.md) for how you can change the below commands to instead
run inference through Nvidia NIM API.

1. Prepare the data

   ```
   python -m nemo_skills.dataset.prepare gsm8k math
   ```

2. Solution augmentation.
   We generate multiple new solutions for each of the original training set problems.

   MATH dataset.

   ```
   ns generate \
       --cluster=slurm \
       --server_type=trtllm \
       --model=/trt_models/llama-3.1-405b-instruct \
       --server_gpus=8 \
       --server_nodes=2 \
       --num_random_seeds=512 \
       --output_dir=/workspace/solution-augmentation/math \
       --eval_args="++eval_type=math" \
       ++dataset=math \
       ++split=train_full \
       ++prompt_config=generic/math-base \
       ++examples_type=math_text_detailed \
       ++prompt_template=llama3-base
   ```

   GSM8K dataset.

   ```
   ns generate \
       --cluster=slurm \
       --server_type=trtllm \
       --model=/trt_models/llama-3.1-405b-instruct \
       --server_gpus=8 \
       --server_nodes=2 \
       --num_random_seeds=64 \
       --output_dir=/workspace/solution-augmentation/gsm8k \
       --eval_args="++eval_type=math" \
       ++dataset=gsm8k \
       ++split=train_full \
       ++prompt_config=generic/math-base \
       ++examples_type=gsm8k_text_detailed \
       ++prompt_template=llama3-base
   ```

3. Problem augmentation.
   We generate new problems using the problems from the training sets as a "seed".

   MATH dataset.

   ```
   ns generate \
       --cluster=slurm \
       --server_type=trtllm \
       --model=/trt_models/llama-3.1-405b-instruct \
       --server_gpus=8 \
       --server_nodes=2 \
       --num_random_seeds=80 \
       --output_dir=/workspace/problem-augmentation/math \
       ++dataset=math \
       ++split=train_full \
       ++prompt_config=generic/problem-augmentation \
       ++examples_type=math_problem_augmentation \
       ++prompt_template=llama3-instruct
   ```

   GSM8K dataset.

   ```
   ns generate \
       --cluster=slurm \
       --server_type=trtllm \
       --model=/trt_models/llama-3.1-405b-instruct \
       --server_gpus=8 \
       --server_nodes=2 \
       --num_random_seeds=10 \
       --output_dir=/workspace/problem-augmentation/gsm8k \
       ++dataset=gsm8k \
       ++split=train_full \
       ++prompt_config=generic/problem-augmentation-similar \
       ++examples_type=gsm8k_problem_augmentation \
       ++prompt_template=llama3-instruct
   ```

4. Solution augmentation for the newly generated problems.
   We generate 32 solutions for each of the new problems.

   We use the Python API in commands below.

   MATH dataset.

   ```python
   from nemo_skills.pipeline import wrap_arguments
   from nemo_skills.pipeline.cli import generate

   # we generated 80 new problems from each original seed problem, so we have a loop
   # to now generate 32 solutions for each of those 80 new data files
   for i in range(80):
       generate(
           cluster="slurm",
           server_type="trtllm",
           model="/trt_models/llama-3.1-405b-instruct",
           server_gpus=8,
           server_nodes=2,
           num_random_seeds=32,
           output_dir=f"/workspace/new-problems-solution-augmentation/math/problem-set{i}",
           ctx=wrap_arguments(
               f"++input_file=/workspace/solution-augmentation/math/generation/output-rs{i} "
               f"++prompt_config=generic/math-base "
               f"++examples_type=math_text_detailed "
               f"++prompt_template=llama3-base "
           ),
       )
   ```

   GSM8K dataset.

   ```python
   from nemo_skills.pipeline import wrap_arguments
   from nemo_skills.pipeline.cli import generate

   # we generated 10 new problems from each original seed problem, so we have a loop
   # to now generate 32 solutions for each of those 10 new data files
   for i in range(10):
       generate(
           cluster="slurm",
           server_type="trtllm",
           model="/trt_models/llama-3.1-405b-instruct",
           server_gpus=8,
           server_nodes=2,
           num_random_seeds=32,
           output_dir=f"/workspace/new-problems-solution-augmentation/gsm8k/problem-set{i}",
           ctx=wrap_arguments(
               f"++input_file=/workspace/solution-augmentation/gsm8k/generation/output-rs{i} "
               f"++prompt_config=generic/math-base "
               f"++examples_type=gsm8k_text_detailed "
               f"++prompt_template=llama3-base "
           ),
       )
   ```

5. Add majority answer as the ground-truth answer.
   Either copy the data locally or run this command on a slurm node.
   You also need to specify the full path to where `/workspace` is mounted
   (we will make it more convenient in the near future by providing the same
   Python/cmdline API as for other scripts).

   ```python
   import subprocess

   # for MATH
   data_folder = "<path to where /workspace is>/new-problems-solution-augmentation/math"
   for i in range(80):
       cmd = (
           f'python -m nemo_skills.evaluation.fill_majority_answer '
           f'    ++input_files="{data_folder}/problem-set{i}/generation/output-rs*.jsonl" '
       )
       subprocess.run(cmd, shell=True, check=True)

   # for GSM8K
   data_folder = "<path to where /workspace is>/new-problems-solution-augmentation/gsm8k"
   for i in range(10):
       cmd = (
           f'python -m nemo_skills.evaluation.fill_majority_answer '
           f'    ++input_files="{data_folder}/problem-set{i}/generation/output-rs*.jsonl" '
       )
       subprocess.run(cmd, shell=True, check=True)
   ```

6. Check for test set contamination.
   We test against GSM8K, MATH, AMC 2023, and AIME 2024.  
 
   Retrieve top-5 similar items from the test sets
   ```
   python -m nemo_skills.inference.retrieve_similar \
      ++retrieve_from="./nemo_skills/dataset/gsm8k/test.jsonl ./nemo_skills/dataset/math/test.jsonl ./nemo_skills/dataset/amc23/test.jsonl ./nemo_skills/dataset/aime24/test.jsonl" \
      ++compare_to="<path to workspace>/new-problems-solution-augmentation/**/output-rs0.jsonl" \
      ++output_file=<path to workspace>/new-problems-solution-augmentation/contamination-retrieved.jsonl \
      ++top_k=5
   ```
   > **_NOTE:_** Currently the above command doesn't run inside docker, so you will need to install additional packages.

   Next, you need to run LLM inference to check those closest found problems from the output file. We use the Llama3.1-405B-Instruct model for this, and here's one way of doing it via Nvidia API catalog.

    ```
    ns check_contamination \
        --cluster=local \
        --input_file=/workspace/new-problems-solution-augmentation/contamination-retrieved.jsonl \
        --output_file=/workspace/new-problems-solution-augmentation/contamination-llm.jsonl \
        --server_type=openai \
        --model=meta/llama-3.1-405b-instruct \
        --server_address=https://integrate.api.nvidia.com/v1 \
        ++check_both_ways=True
    ```
    
   Identify all the problems for which the `contaminated` key has the output True. 
   Add the entry `"contaminated": True` in all the generation files in `<path to workspace>/new-problems-solution-augmentation/`. Here is a sample python script for this:

    ```python
    def load_contaminated_problems(jsonl_file):
        contaminated_problems = set()
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['contaminated']:
                    contaminated_problems.add(data['problem'])
        return contaminated_problems

    def update_output_files(directory, contaminated_problems):
        file_pattern = str(Path(directory) / '**' / 'output-rs*.jsonl')
        for file_path in glob.glob(file_pattern, recursive=True):
            temp_file_path = Path(file_path).with_suffix('.temp')
            
            with open(file_path, 'r') as input_file, open(temp_file_path, 'w') as output_file:
                for line in input_file:
                    data = json.loads(line)
                    if data['problem'] in contaminated_problems:
                        data['contaminated'] = True
                    json.dump(data, output_file)
                    output_file.write('\n')
            
            # Replace the original file with the updated one
            temp_file_path.replace(file_path)
            print(f"Updated file: {file_path}")

    contaminated_problems = load_contaminated_problems("<path to workspace>/new-problems-solution-augmentation/contamination-llm.jsonl")

    update_output_files("<path to workspace>/new-problems-solution-augmentation/", contaminated_problems)

    ``` 



7. Now all the data is generated and you can follow up by converting it to the SFT format.
   We remove the problems marked as contaminated. 
   We also remove solutions with length > 1024 Llama tokens.
   To avoid the models from generating extremely short solutions, we remove solutions shorter than 200 characters.    
   ```
   python -m nemo_skills.training.prepare_sft_data \
      ++prompt_template=llama3-instruct \
      ++prompt_config=generic/math \
      ++input_files="/workspace/solution-augmentation/**/output-rs*.jsonl /workspace/new-problems-solution-augmentation/**/output-rs*.jsonl" \
      ++output_path=/workspace/sft_data.jsonl \
      ++filters.remove_contamindated=true \
      ++filters.remove_len_outlier_solutions=true \
      ++use_chars_for_min_length=true \
      ++min_solution_length=200 \
      ++hf_model_name="meta-llama/Meta-Llama-3.1-8B" \
      ++max_solution_length=1024 \
      ++generation_suffix='"<|eot_id|>"'
   ```


## Model training

We assume you have `/workspace` defined in your cluster config and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

1. Get the data from [HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
   This might take 20-30 minutes (or more depending on your network connection) and will use ~20Gb of RAM.

   ```python
   import json

   from datasets import load_dataset
   from tqdm import tqdm

   dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

   print("Converting dataset to jsonl format")
   output_file = "openmathinstruct2.jsonl"
   with open(output_file, 'w', encoding='utf-8') as f:
       for item in tqdm(dataset):
           f.write(json.dumps(item, ensure_ascii=False) + '\n')

   print(f"Conversion complete. Output saved as {output_file}")
   ```

   You can also download a subset of the data by using e.g. `split='train_5M'` that we used to train 70B model.
   See the dataset page for more details about this.

2. Convert the data into the SFT format that NeMo-Aligner understands.

   ```
    python -m nemo_skills.training.prepare_sft_data \
       ++prompt_template=llama3-instruct \
       ++prompt_config=generic/math \
       ++preprocessed_dataset_files=/workspace/openmathinstruct2.jsonl \
       ++output_key=generated_solution \
       ++output_path=/workspace/openmathinstruct2-sft.jsonl \
       ++filters.drop_multi_boxed=false \
       ++filters.trim_prefix=false \
       ++filters.trim_solutions=false \
       ++filters.drop_incorrect_arithmetic=false \
       ++filters.split_arithmetic=false \
       ++generation_suffix='"<|eot_id|>"';
   ```

3. Download the base model and convert it to NeMo format. The instructions below are for Llama3.1-8B, but the same commands should work for 70B model as well.

   ```
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download meta-llama/Llama-3.1-8B --local-dir Llama-3.1-8B

   ns convert \
       --cluster=local \
       --input_model=/workspace/Llama-3.1-8B \
       --output_model=/workspace/llama3.1-8b-nemo \
       --convert_from=hf \
       --convert_to=nemo \
       --num_gpus=1 \
       --hf_model_name=meta-llama/Llama-3.1-8B
   ```

4. Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
   timeout policy, you can run multiple dependent jobs with `--num_training_jobs=N`.

   ```
   ns train \
       --cluster=slurm \
       --expname=openmathinstruct2-repro-8b \
       --output_dir=/workspace/openmathinstruct2-repro/checkpoints \
       --nemo_model=/workspace/llama3.1-8b-nemo \
       --num_nodes=8 \
       --num_gpus=8 \
       --average_steps 10000 20000 30000 40000 50000 60000 \
       --training_data=/workspace/openmathinstruct2-sft.jsonl \
       ++model.data.train_ds.micro_batch_size=4 \
       ++model.tensor_model_parallel_size=4 \
       ++model.pipeline_model_parallel_size=1 \
       ++model.optim.lr=2e-5 \
       ++trainer.sft.save_interval=10000 \
       ++trainer.sft.max_steps=60000 \
       ++trainer.sft.max_epochs=-1
   ```

   For 70B model, we used 5M data subset and the following parameters, but training
   it longer is likely going to improve results.

   ```
   ns train \
       --cluster=slurm \
       --expname=openmathinstruct2-repro-70b \
       --output_dir=/workspace/openmathinstruct2-repro-70b/checkpoints \
       --nemo_model=/workspace/llama3.1-70b-nemo \
       --num_nodes=32 \
       --num_gpus=8 \
       --average_steps 3330 6660 9990 13320 16650 20000 \
       --training_data=/workspace/openmathinstruct2-sft-5M.jsonl \
       ++model.data.train_ds.micro_batch_size=1 \
       ++model.tensor_model_parallel_size=8 \
       ++model.pipeline_model_parallel_size=2 \
       ++model.optim.lr=1e-5 \
       ++trainer.sft.save_interval=3330 \
       ++trainer.sft.max_steps=20000 \
       ++trainer.sft.max_epochs=-1
   ```

   If you have a job timeout, it's necessary to set the maximum time per run to 40 minutes
   before the timeout to allow for the final checkpoint to be saved. E.g. if your timeout is 4 hours,
   add `++exp_manager.max_time_per_run=00:03:20:00`


If you want to follow up with checkpoint conversion and evaluation, see
[training docs](/docs/training.md#python-api) for an example of how to do it
through a convenient Python API.


## Dataset contamination explorer

To reproduce our dataset contamination explorer demo refer to [dataset_explorer_demo/README.md](/dataset_explorer_demo/README.md)