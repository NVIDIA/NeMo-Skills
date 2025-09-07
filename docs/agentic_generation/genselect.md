# GenSelect


GenSelect is a generative Best-of-N method we introduced in the [OpenReasoning paper](https://arxiv.org/abs/2504.16891) followed by a more focused paper -- [GenSelect: A Generative Approach to Best-of-N](https://arxiv.org/abs/2507.17797). The method essentially uses an LLM to reason over and select the best candidate solution among the N candidates, leveraging LLMs' comparative strengths while scaling efficiently across parallel sampling budgets.


## Usage

We support GenSelect via the [generation pipeline](https://nvidia.github.io/NeMo-Skills/pipelines/generation/). To use genselect, just pass in `++genselect=True` when using the generation/eval pipelines.
We support both offline and online GenSelect:

- Offline mode: The N solutions/trajectories have already been generated and can be specified via `++genselect_config.generation_dir=<PATH_TO_GENERATED_DIR>`
- Online mode: The N solutions need to be generated as part of the generation job.


## Key Parameters

The GenSelect pipeline uses the same inference parameters as the generate pipeline. We allow overriding of three key inference config params:

- `temperature`: Inference temperature
- `tokens_to_generate`: Inference token budget
- `prompt_config`: Default config is `generic/genselect`

Other inference params like `top_p`, `min_p`, etc., inherit from the main generation configuration.

Next, we discuss GenSelect specific params. We first discuss parameters common to both the online and offline mode, and then discuss paramters specific to the offline mode.

### Common Parameters
- `window_size`: Number of solutions compared in a single GenSelect comparison (typically set to 8). Consider your model's context window size when setting this value (or allow for soft failure via `++server.enable_soft_fail=True`).
- `comparison_key`: The key from the generation output used for comparison (default: `generation`)

### Offline GenSelect Parameters
- `generation_dir`: The directory where the *offline* generated solutions are stored. We assume the solutions to be in `output-rs*.jsonl` files.
- `num_initial_solutions`: Number of solutions from the offline generated solutions that are used for GenSelect.


To specify these variables, say `window_size=16`, pass `++genselect_config.window_size=16` to the generate/eval pipelines.


## Sample Commands


### Online GenSelect

In this example we show how to







## Papers

- [GenSelect: A Generative Approach to Best-of-N](https://arxiv.org/abs/2507.17797)
- [AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset](https://arxiv.org/abs/2504.16891)


If you find GenSelect useful, please consider citing us!

```bibtex

@article{toshniwal2025genselect,
  title={{GenSelect: A Generative Approach to Best-of-N}},
  author={Shubham Toshniwal and Ivan Sorokin and Aleksander Ficek and Ivan Moshkov and Igor Gitman},
  year={2025},
  journal = {arXiv preprint arXiv:2507.17797},
}

@article{moshkov2025aimo2,
  title   = {{AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset}},
  author  = {Ivan Moshkov and Darragh Hanley and Ivan Sorokin and Shubham Toshniwal and Christof Henkel and Benedikt Schifferer and Wei Du and Igor Gitman},
  year    = {2025},
  journal = {arXiv preprint arXiv:2504.16891}
}
```