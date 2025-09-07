# GenSelect


GenSelect is a generative Best-of-N method we introduced in the [OpenReasoning paper](https://arxiv.org/abs/2504.16891) followed by a more focused paper -- [GenSelect: A Generative Approach to Best-of-N](https://arxiv.org/abs/2507.17797). The method essentially uses an LLM to reason over and select the best candidate among the N candidates, leveraging LLMs' comparative strengths while scaling efficiently across parallel sampling budgets.


## Usage

We support GenSelect via the [generation pipeline](https://nvidia.github.io/NeMo-Skills/pipelines/generation/). To use genselect, just pass in `++genselect=True` when using the generation/eval pipelines.
We support both offline and online GenSelect:

- Offline mode: The N trajectories have already been generated and can be specified via `++genselect.generation_dir=<PATH_TO_GENERATED_DIR>`
- Online mode: The N trajectories need to be generated as part of the generation job.



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