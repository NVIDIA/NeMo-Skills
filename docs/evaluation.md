# Model evaluation

## Quick start

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.
Please note that ~1% difference in accuracy is expected when running inference on
different GPU types or with different inference frameworks.

1. Download one of [our models](https://huggingface.co/collections/nvidia/openmath-65c5619de2ba059be0775014) or get some other checkpoint.
2. [Convert the model to the right format](/docs/checkpoint-conversion.md) if required.
3. Run the evaluation (assuming one of our finetuned models, nemo inference, gsm8k greedy decoding)

   ```
   python pipeline/run_eval.py \
     --model_path <path to .nemo> \
     --server_type nemo \
     --output_dir ./test-results \
     --benchmarks gsm8k:0 \
     --num_gpus <number of GPUs on your machine/cluster node> \
     --num_nodes 1 \
     +prompt=code_sfted \
     ++prompt.few_shot_examples.num_few_shots=0 \
     ++split_name=test
   ```

   If you want to evaluate a model that was not finetuned through our pipeline, but still
   allow it to use Python interpreter, you can show it a couple of few-shot examples

   ```
   +prompt=code_base \
   ++prompt.few_shot_examples.examples_type=gsm8k_text_with_code \
   ++prompt.few_shot_examples.num_few_shots=5
   ```

   If you need to, change the batch size with `batch_size=<X>` argument.

4. Compute metrics

   ```
   python pipeline/compute_metrics.py \
     --prediction_jsonl_files ./test-results/gsm8k/output-greedy.jsonl
   ```

   If you evaluated multiple benchmarks or used multiple samples per benchmark, you can also run the following script
   to summarize all available metrics.

   ```
   python pipeline/summarize_results.py ./test-results
   ```

Read on to learn details about how evaluation works!

## Details

Let's break down what [pipeline/run_eval.py](/pipeline/run_eval.py) is doing.

- Starts a local [sandbox](/docs/sandbox.md) which will handle code execution requests.
- Starts an LLM server in a docker container (defined in the `NEMO_SKILLS_CONFIG` file).
- Waits for the sandbox and server to start.
- Runs [nemo_skills/inference/generate_solutions.py](/nemo_skills/inference/generate_solutions.py) to
  generate solutions for all benchmarks requested (potentially running multiple samples per benchmark).
- Runs [nemo_skills/evaluation/evaluate_results.py](/nemo_skills/evaluation/evaluate_results.py) on each
  of the generated output files.
- If running in a Slurm cluster, you can parallelize evaluation across multiple nodes. You can also
  customize any of the parameters of evaluation - all extra arguments of the
  run_eval.py will be passed directly to the generate_solutions.py script.

Here is an example of how to manually reproduce the call to run_eval.py script from
the [quick start](#quick-start) section.

1. Start a sandbox. This will block your shell, so either run in the background or make sure you can open another shell on the same machine:

   ```
   ./nemo_skills/code_execution/local_sandbox/start_local_sandbox.sh
   ```

   Get the IP of the sandbox by running

   ```
   docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' `docker ps -a | grep local-sandbox  | awk '{print $1}'`
   ```

2. Start an LLM server. The commands differ based on the server type. Here is an example for starting NeMo-based inference server.
   Make sure to run this from the root of the repository. Same as above, this will block your shell.

   ```
   docker run --rm --gpus all --ipc=host -v `pwd`:/code -v <path to the .nemo model>:/model igitman/nemo-skills-sft:0.2.0 \
   bash -c 'PYTHONPATH=/code python /code/nemo_skills/inference/server/serve_nemo.py \
     gpt_model_file=/model \
     trainer.devices=<number of GPUs> \
     tensor_model_parallel_size=<number of GPUs> \
     ++sandbox.host=<Sandbox IP from the step above>'
   ```

   Wait until you see "Running on <ip address>" message and make a note of this IP.

   If you want to use TensorRT-LLM server instead, you can run the following command

   ```
   docker run --rm --gpus all --ipc=host -v `pwd`:/code -v <path to the trtllm model>:/model igitman/nemo-skills-trtllm:0.2.0 \
   bash -c 'export PYTHONPATH=/code && \
   mpirun -n <number of GPUs> --allow-run-as-root python /code/nemo_skills/inference/server/serve_trt.py --model_path=/model'
   ```

3. Run the generation command. Customize as necessary (running with `--help` will show the details)

   ```
   python nemo_skills/inference/generate_solutions.py \
     output_file=./test-results/gsm8k/output-greedy.jsonl \
     +prompt=code_sfted \
     ++prompt.few_shot_examples.examples_type=null \
     ++prompt.few_shot_examples.num_few_shots=0 \
     ++prompt.context_type=empty \
     ++dataset=gsm8k \
     ++split_name=test \
     ++server.server_type=nemo \
     ++server.host=<IP from the step above> \
     ++sandbox.host=<Sandbox IP from the sandbox launch step>
   ```

4. Run the evaluation command. Note that you need to provide a sandbox IP, because evaluation is running in the sandbox.

   ```
   python nemo_skills/evaluation/evaluate_results.py \
     prediction_jsonl_files=./test-results/gsm8k/output-greedy.jsonl \
     ++sandbox.host=<Sandbox IP>
   ```

After this you would typically follow up with the same command to compute metrics as in the [quick start](#quick-start).


## Typical customizations

To customize the prompt template for the model, create a new .yaml file inside
[nemo_skills/inference/prompt](/nemo_skills/inference/prompt) folder. Have a look
at the existing templates there for an example.

You can run `python nemo_skills/inference/generate_solutions.py --help`
to see other available customization options.