# How to build all necessary dockerfiles

Some dockerfiles are directly included in this folder and for some other the instructions on building them are below.
To build one of the existing dockerfiles use a command like this

```
docker build -t igitman/nemo-skills-nemo:0.5.0 -f dockerfiles/Dockerfile.nemo .
```
It might take a long time for some of the images.

## Building trtllm image

Follow instructions in [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step).

Our current container is built from `v0.18.1` code version.

## Building vllm image

It's possible to directly re-use the latest vLLM docker image. The only other change
we do is to add OpenRLHF (custom fork) into it. To do that, follow these steps

1. `git clone https://github.com/Kipok/OpenRLHF`
2. checkout tag/commit
3. run `pip install -e .`

Current vllm docker version: vllm/vllm-openai:v0.8.3

Current Kipok/OpenRLHF version: 9001bc7026517e8f51682978c52cc41eb1d2c563

## Building sglang image

Currently we can directly reuse latest sglang docker image.

```
cd /sgl-workspace/sglang/
git apply <path to NeMo-SKills>/dockerfiles/sglang.patch
```

then run `docker ps -a` and note image id of your running container. Do `docker commit <image id>`
and `docker tag <printed hash> igitman/nemo-skills-sglang:0.5.0` and push that image.

Current sglang docker version: lmsysorg/sglang:v0.4.5-cu118
