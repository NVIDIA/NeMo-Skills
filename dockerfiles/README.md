# How to build all necessary dockerfiles

All dockerfiles are directly included in this folder, except vllm and sglang.
To build one of the existing dockerfiles use a command like this

```
docker build -t igitman/nemo-skills-nemo:0.5.0 -f dockerfiles/Dockerfile.nemo .
```
It might take a long time for some of the images.

## Building vllm image

To build vllm image do the following

1. git clone https://github.com/vllm-project/vllm
2. checkpoint tag/commit, e.g. `git checkout v0.7.1`
3. remove the check for wheel size https://github.com/vllm-project/vllm/blob/main/Dockerfile#L130
4. check if vllm installation part is built from "-devel" image. If not, change that (otherwise nvcc will be missing)
5. build a docker with `docker build -t igitman/nemo-skills-vllm:0.5.0 -f Dockerfile .`
6. add OpenRLHF (you can include it in the previous step by modifying dockerfile or do it after and then commit the docker)
   1. `git clone https://github.com/OpenRLHF/OpenRLHF`
   2. checkpoint tag/commit
   3. remove all hardcoded versions from requirements.txt to get latest
   4. run `pip install -e .`

Current vllm version: v0.7.0

Current OpenRLHF version: 631b0bcb7c14ec2fd4117a43a661903ed60d26cc

## Building sglang image

Currently we can directly reuse `lmsysorg/sglang:latest` image except we need to apply the patch included in this folder.
To do that launch a docker, then run

```
cd /sgl-workspace/sglang/
git apply <path to NeMo-SKills>/dockerfiles/sglang.patch
```

then run `docker ps -a` and note image id of your running container. Do `docker commit <image id>`
and `docker tag <printed hash> igitman/nemo-skills-sglang:0.5.0` and push that image.

Current sglang version: lmsysorg/sglang:v0.4.2.post2-cu125