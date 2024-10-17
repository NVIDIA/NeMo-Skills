# Common parameters

Many of our scripts have a shared set of common parameters that we list here.

## All pipeline scripts

All scripts inside pipeline folder have the following parameters.

- **--cluster**: You always need to specify a cluster config that will be used to
  control where the job is executed.
- **--config_dir**: By default we search for cluster configs inside `cluster_configs`
  local folder, but you can control where they are located with this parameter.
  You can also use `NEMO_SKILLS_CONFIG_DIR` environment variable for this purpose.
- **--log_dir**: Can be used to customize the location of slurm logs.
- **--expname**: You can always specify an experiment name, which is a
  [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) concept. This will control where
  the metadata is stored, the slurm job name and allows you to chain jobs one
  after the other using the `--run_after` argument.
- **--run_after**: Can be used in conjunction with `--expname` to chain jobs to
  run one after another (only applicable on slurm). E.g. run training job with
  `--expname my-training-run` and then launch an eval with `--run_after my-training-run`.
- **--partition**: Can be used to run in a specific slurm partition (e.g. commonly used
  to launch interactive jobs).

## Generation scripts

All of the scripts that involve LLM data generation accept a common set of parameters.

- **--model**: Either path to the model file or an API model name.
- **--server_type**: `nemo`, `trtllm`, `vllm` or `openai`. This is used on the client side
  to correctly format a request to a particular server. This needs to match model
  checkpoint format if self-hosting the model or has to be `openai` for both Nvidia
  NIM API as well as the OpenAI API.
- **--server_address**: Only relevant for API models. E.g. use
  `https://integrate.api.nvidia.com/v1` for Nvidia API and
  `https://api.openai.com/v1` for OpenAI API.
- **--server_gpus**: Number of GPUs needed to host a model (only applicable to self-hosted models).
- **--server_nodes**: Number of nodes needed to host a model (only applicable to self-hosted models).
- **--server_args**: Any other arguments you need to pass to a corresponding server.
  E.g. use `--server_args="--gpu-memory-utilization=0.99"` to change gpu memory utilization of a
  vLLM server.