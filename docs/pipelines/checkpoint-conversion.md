# Checkpoint conversion

!!! info

    This pipeline starting script is [nemo_skills/pipeline/convert.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/pipeline/convert.py)

    All extra parameters are passed to one of the following scripts

    * For conversion to NeMo:
        - Default: [nemo_skills/conversion/hf_to_nemo.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/conversion/  hf_to_nemo.py)
        - If `--model_type=qwen`: [nemo_skills/conversion/hf_to_nemo_qwen.py](https://github.com/Kipok/NeMo-Skills/blob/main/  nemo_skills/conversion/hf_to_nemo_qwen.py)

    * For conversion to TensorRT-LLM:
        - Default: [nemo_skills/conversion/hf_to_trtllm.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/conversion/  hf_to_trtllm.py)
        - If `--model_type=qwen`: [nemo_skills/conversion/hf_to_trtllm_qwen.py](https://github.com/Kipok/NeMo-Skills/blob/main/  nemo_skills/conversion/hf_to_trtllm_qwen.py)

    * For conversion to HuggingFace: [nemo_skills/conversion/nemo_to_hf.py](https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/conversion/nemo_to_hf.py)


We support 3 common model formats. Here are some recommendations on when each format should be used.

- [HuggingFace (via vLLM)](https://github.com/vllm-project/vllm).
  If you want to run a small-scale generation quickly
  or play with models, it's most convenient to use HF format directly via a vllm server.

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
  If you want to run a large-scale generation, it's highly recommended to use TensoRT-LLM format.
  The time it takes to convert the checkpoint will be more than offset by a much faster generation
  than either vLLM or NeMo.

- [NeMo](https://github.com/NVIDIA/NeMo).
  NeMo is the only supported format for training, so you need to use it with the
  [training pipeline](training.md). We don't recommend running inference in NeMo
  as it is much slower than both vLLM and TensorRT-LLM servers.

To convert the checkpoint from one format to another use a command like this

```bash
ns convert \
    --cluster=slurm \
    --input_model=/hf_models/Meta-Llama-3.1-70B-Instruct \
    --output_model=/trt_models/llama3.1-70b-instruct \
    --convert_from=hf \
    --convert_to=trtllm \
    --num_gpus=8 \
    --hf_model_name=meta-llama/Meta-Llama-3.1-70B-Instruct
```

You can provide any extra arguments that will be passed directly to the underlying conversion scripts.
Here are a few things to keep in mind

- We currently only support Llama-based and Qwen-based models (enable with `--model_type qwen`). The other kinds
  of models are most likely easy to add, we just didn't have a use-case for them yet (please open an issue if the
  model you want to use is not supported).
- You cannot convert from trtllm format, only to it.
- You cannot convert from nemo to trtllm directly and need to do it in 2 stages, to nemo->hf and then hf->trtllm.
- Please check [NeMo](https://github.com/NVIDIA/NeMo) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
  documentation to learn best recommended parameters for converting each specific model.
