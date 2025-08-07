# How to build all necessary dockerfiles

Some dockerfiles are directly included in this folder and for some others the instructions to build them are below.
To build one of the existing dockerfiles use a command like this

```
docker build -t igitman/nemo-skills-nemo:0.7.0 -f dockerfiles/Dockerfile.nemo .
```
It might take a long time for some of the images.

## Building trtllm image

We directly use official TensorRT-LLM ngc containers. Current version is `nvcr.io/nvidia/tensorrt-llm/release:0.21.0`.
