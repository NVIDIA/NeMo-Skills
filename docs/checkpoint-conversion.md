# Checkpoint conversion

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.

## HuggingFace to TensorRT-LLM

The instructions to convert from HuggingFace to TensorRT-LLM checkpoint format
can be found in [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM/) (they are model-specific).
Since we mostly work with Llama-based models (Llama2, CodeLlama, Mistral), we put the corresponding
conversion script in this repo for convenience. But please refer to the original repo for any other model's conversion.

Convert the model in 3 steps. Note that you need to explicitly specify input/output length, batch size and number of GPUs.
Make sure to run the commands inside TensorRT-LLM docker container, e.g. you can use `igitman/nemo-skills-trtllm:0.1.0`

```
python nemo_skills/conversion/hf_to_trtllm.py \
    --model_dir <path to the HF folder> \
    --output_dir <tmp file for trtllm checkpoint> \
    --dtype <float16, bfloat16, float32> \
    --tp_size <number of GPUs>

trtllm-build \
    --checkpoint_dir <tmp file for trtllm checkpoint> \
    --output_dir <final path for the trtllm checkpoint> \
    --gpt_attention_plugin <dtype from step above> \
    --gemm_plugin <dtype from step above> \
    --context_fmha <"enable" on A100+ GPUs and "disable" otherwise> \
    --paged_kv_cache <"enable" on A100+ GPUs and "disable" otherwise> \
    --max_input_len 4096 \
    --max_output_len 512 \
    --max_batch_size <desired batch size>

cp <path to the HF folder>/tokenizer.model <final path for the trtllm checkpoint>/tokenizer.model
```

Please note that these are just example parameters and we refer you to the
[TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/) to learn
how to configure conversion for your exact use-case.

## NeMo to HuggingFace

Most of the conversion scripts from NeMo to HuggingFace format can be found in
[NeMo repository](https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling).
Since we mostly work with Llama-based models (Llama2, CodeLlama, Mistral), we put the corresponding
conversion script in this repo for convenience. But please refer to the original repo for any other model's conversion.

To convert Llama-based model you can use the following command.
Make sure to run it inside NeMo docker container, e.g. you can use `igitman/nemo-skills-sft:0.1.0`

```
python nemo_skills/conversion/nemo_to_hf.py \
    --in-path <nemo path> \
    --out-path <where to save HF checkpoint> \
    --hf-model-name <HF name of the reference model, e.g. codellama/CodeLlama-7b-Python-hf> \
    --precision bf16 \
    --max-shard-size 10GB
```

## HuggingFace to Nemo

Most of the conversion scripts from HuggingFace to NeMo format can be found in
[NeMo repository](https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling).
Since we mostly work with Llama-based models (Llama2, CodeLlama, Mistral), we put the corresponding
conversion script in this repo for convenience. But please refer to the original repo for any other model's conversion.

To convert Llama-based model you can use the following command.
Make sure to run it inside NeMo docker container, e.g. you can use `igitman/nemo-skills-sft:0.1.0`

```
python nemo_skills/conversion/hf_to_nemo.py \
    --in-path <HF folder path> \
    --out-path <where-to-save.nemo> \
    --precision bf16
```

## NeMo to TensorRT-LLM

The support for direct conversion from NeMo to TensorRT-LLM is coming soon! For now you can do NeMo -> HuggingFace -> TensorRT-LLM instead.