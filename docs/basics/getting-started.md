# Getting Started

Let's walk through a little tutorial to get started working with nemo-skills.

We will use a simple generation job to run LLM inference in different setups (through API, hosting model
locally and on slurm cluster). This will help you understand some important concepts we use (e.g.
[cluster configs](./cluster-configs.md)) as well as to setup your machine to run any other jobs.

## Setup

First, let's install nemo-skills

```bash
pip install git+https://github.com/NVIDIA/NeMo-Skills.git
```

or if you have the repo cloned locally, you can run `pip install -e .` instead.

Now, let's create a simple file with just 3 data points that we want to run inference on

```jsonl title='input.jsonl'
{"prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
{"prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Awful"}
{"prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
```

save the above into `./input.jsonl`.

Let's also create a [prompt config](../basics/prompt-format.md) that defines how input data is combined into an LLM prompt

```yaml title='prompt.yaml'
system: "When answering a question always mention NeMo-Skills repo in a funny way."

user: |-
   Question: {prompt}
   Option A: {option_a}
   Option B: {option_b}
```

save the above into `./prompt.yaml`.

## API inference

Now we are ready to run our first inference. Since we want to use API models, you need to have an API key.
You can either use [OpenAI models](https://platform.openai.com/docs/overview) or
[Nvidia NIM models](https://build.nvidia.com/) (just register there and you will get some free credits to use!).

=== "Nvidia NIM models"

    ```bash
    export NVIDIA_API_KEY=<your key>
    ns generate \
        --server_type=openai \
        --model=meta/llama-3.1-8b-instruct \
        --server_address=https://integrate.api.nvidia.com/v1 \
        --output_dir=./generation \
        ++input_file=./input.jsonl \
        ++prompt_config=./prompt.yaml
    ```

=== "OpenAI models"

    ```bash
    export OPENAI_API_KEY=<your key>
    ns generate \
        --server_type=openai \
        --model=gpt-4o-mini \
        --server_address=https://api.openai.com/v1 \
        --output_dir=./generation \
        ++input_file=./input.jsonl \
        ++prompt_config=./prompt.yaml
    ```

You should be able to see a jsonl file with 3 lines containing the original data and a new `generation` key
with an LLM output for each prompt.

## Local inference

If you pay attention to the log of above commands, it's going to print you this warning

```
WARNING  Cluster config is not specified. Running locally without containers. Only a subset of features is supported and you're responsible for installing any required dependencies. It's recommended to run `ns setup` to define appropriate configs!
```

...

# Important details

Let us summarize a few details that are important to keep in mind when using nemo-skills.

**Using containers**. Most nemo-skills commands require using multiple docker containers that communicate with each
other. The containers used are specified in your [cluster config](./cluster-configs.md) and we will start them
for you automatically. But it's important to keep this in mind since e.g. any packages that you install
aren't going to be available for nemo-skills jobs unless you change the containers. This is also the reason why
we have a `mounts` section in the cluster config and all paths that you specify in various commands need to reference
the *mounted* path, not your local/cluster path. Another important implication is that any environment variables
are not accessible to our jobs by default and you need to explicitly list then in your cluster configs.

**Code packaging**. All nemo-skills commands will *package* your code to make it available in container or in slurm jobs.
This means that your code will be copied to `~/.nemo_run/experiments` folder locally or `job_dir` (defined in your
[cluster config](./cluster-configs.md)) on cluster. All our commands accept `expname` parameter and the code and other
metadata will be available inside `expname` subfolder. We will always package any git repo you're running nemo-skills
commands from, as well as the nemo-skills Python package and they will be available inside docker/slurm under `/nemo_run/code`.
You can read more in [code packaging](./code-packaging.md) documentation.

**Running commands**. Any nemo-skills command can be accessed via `ns` command-line as well as through Python API.
It's important to keep in mind that all arguments to such commands are divided into *wrapper* arguments (typically
used as `--arg_name`) and *main* arguments (typically specified as `++arg_name` since we use
[Hydra](https://hydra.cc/) for most scripts). The *wrapper* arguments configure the job itself (such as where to run it
or how many GPUs to request in slurm) while the *main* arguments are directly passed to whatever underlying script the
wrapper command calls. When you run `ns <command> --help`, you will always see the *wrapper* arguments displayed directly
as well as the information on what actual script is used underneath and an extra command you can run to see
what *inner* arguments are available. You can learn more about this in [running commands](./running-commands.md) documentation.

**Scheduling slurm jobs**. Our code is primarily built to schedule jobs on slurm clusters and that affects many design decisions
we made. A lot of the arguments for nemo-skills commands are only used with slurm cluster configs and are ignored when
running "local" jobs. It's important to keep in mind that the recommended way to submit slurm jobs is from a *local*
workstation by defining `ssh_tunnel` section in your [cluster config](./cluster-configs.md). This helps us avoid
installing nemo-skills and its dependencies on the clusters and makes it very easy to switch between different slurm clusters
and a local "cluster" with just a single `cluster` parameter.
