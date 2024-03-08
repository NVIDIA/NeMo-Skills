# Supervised finetuning

Supervised finetuning (SFT) is the final stage of our pipeline. We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/)
to run SFT and would encourage you to check their documentation to learn more details.
Here are the commands to prepare data, run SFT and evaluate the finetuned models.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

1. Prepare the dataset, if you've [generated new solutions](/docs/synthetic-data-generation.md).
   E.g. for gsm8k train_full subset (which combines our custom train-validation split together)

   ```
   python nemo_skills/finetuning/prepare_sft_data.py \
       ++prediction_jsonl_files=<path to the generated synthetic data>/output-rs*.jsonl \
       ++output_path=sft-data.jsonl
   ```

   Note that `prediction_jsonl_files` can accept multiple glob patterns separated by space.

2. Run SFT + checkpoint averaging + evaluation in one script.

   ```
   python pipeline/run_sft_and_eval.py \
      --expname <name for experiment> \
      --nemo_model <path to the nemo model> \
      --num_nodes <number of nodes> \
      --num_gpus <number of GPUs per node> \
      ++model.data.train_ds.file_path=/data/<path to the data inside NEMO_SKILLS_DATA folder>
   ```

   This will put all checkpoints, results and logs inside `$NEMO_SKILLS_RESULTS` folder.
   Note that you can provide `--stages` argument to control which steps to run. E.g.
   to skip evaluation use `--stages sft prepare_eval` or to only run evaluation
   (e.g. to re-run with different sampling parameters) use `--stages eval`.

   You can also customize any evaluation parameters with `--extra_eval_args`, e.g.
   to use 2 evaluation nodes, batch size of 32 and evaluate on the test set use

   ```
   --extra_eval_args="--num_nodes=2 ++split_name=test ++batch_size=32 "
   ```

   You can customize any of the SFT parameters by directly providing those
   arguments to the [pipeline/run_sft_and_eval.py](/pipeline/run_sft_and_eval.py) script (training data is already customized
   in the example above). E.g. to disable dropout and use tensorboard logging instead of wandb you can set

   ```
   --disable_wandb \
   ++model.ffn_dropout=0.0 \
   ++model.attention_dropout=0.0 \
   ++model.hidden_dropout=0.0
   ```

Alternatively, you can run all the steps separately.

1. Run SFT

   ```
   python pipeline/run_sft.py \
      --expname <name for experiment> \
      --checkpoints_folder <where to save checkpoints>
      --nemo_model <path to the nemo model> \
      --num_nodes <number of nodes> \
      --num_gpus <number of GPUs per node> \
      ++model.data.train_ds.file_path=/data/<path to the data inside NEMO_SKILLS_DATA folder>
   ```

   Note that you cannot submit multiple dependent jobs with this script and would have to do this manually if required.

2. Run checkpoint averaging

   ```
   python pipeline/prepare_eval.py \
       --training_folder <path to the checkpoints folder above>/training/checkpoints \
       --output_path <where to place the averaged model> \
       --nemo_model <same as in the run sft step, needed to get config>
   ```

3. Run [evaluation](/docs/evaluation.md)
