# Prerequisites and Getting Started

## Installation

To get started first install the repo (python 3.10+). Either clone and run `pip install -e .` or install directly with

```bash
pip install git+https://github.com/NVIDIA/NeMo-Skills.git
```

## Environment variables

Depending on which pipelines you run, you might need to define the following environment variables

``` bash
# only needed for training (can opt-out with --disable_wandb)
export WANDB_API_KEY=...
# only needed if using gated models, like llama3.1
export HF_TOKEN=...
# only needed if running inference with OpenAI models
export OPENAI_API_KEY=...
# only needed if running inference with Nvidia NIM models
export NVIDIA_API_KEY=...
```

## Preparing data

If you want to run evaluation or use training datasets of popular benchmarks (e.g. math/gsm8k) for data augmentation,
you need to run the following commands to prepare the data.

```bash
python -m nemo_skills.dataset.prepare
```

If you're only interested in a subset of datasets (e.g. only math-related or code-related), run with
`--dataset_groups ...` and if you only need a couple of specific datasets, list them directly e.g.

```bash
python -m nemo_skills.dataset.prepare gsm8k human-eval mmlu ifeval
```

If you have the repo cloned locally, the data files will be available inside `nemo_skills/dataset/<benchmark>/<split>.jsonl`
and if you installed from pip, they will be downloaded to wherever the repo is installed, which you can figure out by running

```bash
python -c "import nemo_skills; print(nemo_skills.__path__)"
```

## Cluster configs

All of the [pipeline scripts](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/pipeline) accept `--cluster` argument which you can use
to control where the job gets executed. That argument picks up one of the configs inside your local
[cluster_configs](https://github.com/NVIDIA/NeMo-Skills/tree/main/cluster_configs)
folder by default, but you can specify another location with `--config_dir` or set it in `NEMO_SKILLS_CONFIG_DIR` env variable.
You can also use `NEMO_SKILLS_CONFIG` env variable instead of the `--cluster` parameter.
The cluster config defines an executor (local or slurm), mounts for data/model access and (slurm-only) various parameters
such as account, partition, ssh-tunnel arguments and so on.

### Environment variables

You can define environment variables in the cluster config file, which will be set inside the container.

```yaml
env_vars:
  - MYENVVAR  # will pick the value from env
  - MYENVVAR2=my_value  # will use my_value
```

If an environment variable is required, and the user must provide it, you can use `required_env_vars` instead. One thing to note is that `required_env_vars` does not support passing values directly, so you must provide them via environment variable only.

## Code packaging

We use [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) for managing our experiments with local and slurm-based
execution supported (please open an issue if you need to run our code on other kinds of clusters).
This means that even if you need to submit jobs on slurm, you do it from your local machine by defining an
appropriate cluster config and nemo-run will package and upload your code, data and manage
all complexities of slurm scheduling. Check their documentation to learn how to fetch logs, check status,
cancel jobs, etc.

To decide which code to package we use the following logic:

1. If you run commands from inside a cloned NeMo-Skills repository, we will package that repository.
2. If you run commands from inside a git repository which is not NeMo-Skills (doesn't have `nemo_skills` top-level folder),
   we will package your current repository and also include `nemo_skills` subfolder from it's installed location.
3. If you run commands from outside of any git repository, we will only package `nemo_skills` subfolder from it's installed
   location.

Put simply, we will always include `nemo_skills` and might optionally include your personal git repository if you're
running commands from it.

!!! note

    When packaging a git repository, NeMo-Run will only package the code tracked by git
    (as well as all jsonl files from `nemo_skills/dataset`).
    Any non-tracked files will not be automatically available inside the container or uploaded to slurm.

    When packaging `nemo_skills` form its installed location (which might not be a git repository), we will
    upload **all** the files inside `nemo_skills` subfolder. Make sure you do not store any heavy files there
    to avoid uploading large files on the cluster with each experiment!

Finally, it's important to keep in mind that whenever you submit a new experiment, NeMo-Run will create a copy of your
code package both locally (inside `~/.nemo_run`) and on cluster (inside `ssh_tunnel/job_dir` path in your cluster config).
If you submit multiple experiments from the same Python script, they will all share code, so only one copy will be
created per run of that script. Even so, at some point, the code copies will be accumulated and you will run out of
space both locally and on cluster. There is currently no automatic cleaning, so you have to monitor for that and
periodically remove local and cluster nemo-run folders to free up space. There is no side effect of doing that (they will
be automatically recreated) as long as you don't have any running jobs when you remove the folders.
If you want to have more fine-grained control over code reuse, you can directly specify `--reuse_code_exp` argument when submitting jobs


## Running pipelines

All of the [pipeline scripts](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/pipeline) can be called in 3 equivalent ways.
As an example let's see how to run [evaluation](../pipelines/evaluation.md) on 10 samples from gsm8k and math benchmarks

```bash title="ns command-line entrypoint"
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=gsm8k:0,math:0 \
    --output_dir=/workspace/test-eval \
    ++max_samples=10
```

```bash title="calling python module directly"
python -m nemo_skills.pipeline.eval \
    --cluster=local \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=gsm8k:0,math:0 \
    --output_dir=/workspace/test-eval \
    ++max_samples=10
```


```python title="using python api"
from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import eval

eval(
    cluster="local",
    server_type="openai",
    model="meta/llama-3.1-8b-instruct",
    server_address="https://integrate.api.nvidia.com/v1",
    benchmarks="gsm8k:0,math:0",
    output_dir="/workspace/test-eval",
    # arguments of the underlying script need to be wrapped
    # you can separate multiple arguments with space or newline
    ctx=wrap_arguments("++max_samples=10"),
)
```

You can also chain multiple pipelines together to set proper slurm dependencies using `--run_after` parameter.
See an example in [training documentation](../pipelines/training.md#chaining-pipelines-with-python).

### Pipeline arguments

We use [Hydra](https://hydra.cc/docs/1.3/intro/) for most of the scripts, so
it's a good idea to read through their documentation if that's the first time you see it.

Most of our pipeline scripts use a mix of normal command-line arguments and Hydra style config overrides
(usually formatted as `++arg_name`). Whenever you
see this, it means that the regular `--arg_name` parameters are used to control the wrapper script itself and
all other parameters are directly passed into the underlying `nemo_skills/...` script called by the wrapper.

See [common parameters](common-parameters.md) for a list of parameters common to all scripts.

## Local execution

To run scripts locally we use docker containers, so make sure you have
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
set up on your machine.

All of our scripts assume that data or models are mounted inside the appropriate container so before running any
commands make sure to modify
[cluster_configs/example-local.yaml](https://github.com/NVIDIA/NeMo-Skills/tree/main/cluster_configs/example-local.yaml).
It's convenient to rename it to local.yaml (so you can use `--cluster local`) after you defined necessary mounts.

Most of our containers are quite heavy, so the first time you run a job that requires a large container, it will take
a while to pull it. You can manually run `docker pull <container>` for all containers defined in the local config
to cache them.

## Slurm jobs

If you're running on slurm, you need to define some additional information inside cluster config.

Populate the commented out fields inside
[cluster_configs/example-slurm.yaml](https://github.com/NVIDIA/NeMo-Skills/tree/main/cluster_configs/example-slurm.yaml).
It's convenient to rename it to slurm.yaml (so you can use `--cluster slurm`) or a cluster name if you use multiple slurm clusters.
