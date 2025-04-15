# Running arbitrary commands

!!! info

    This pipeline starting script is [nemo_skills/pipeline/run_cmd.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/run_cmd.py)

    All extra parameters are directly executed as a shell command.

We often need to run arbitrary pre/post processing commands as part of a larger pipeline and thus we provide a simple
`run_cmd` utility that can be used to schedule those on slurm. Here is an example that simply enters the packaged
code and tries to install it (will finish with error if not running from NeMo-Skills repo or other installable package).

```bash
ns run_cmd --cluster=local cd /nemo_run/code/ && pip install -e .
```

There are many more examples of how to use `run_cmd` throughout our documentation.