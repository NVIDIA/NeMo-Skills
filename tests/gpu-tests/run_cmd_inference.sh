# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
# model needs to be inside /mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
# if you need to place it in a different location, modify test-local.yaml config
# example: HF_TOKEN=<> ./tests/gpu-tests/run.sh
set -e

export NEMO_SKILLS_TEST_HF_MODEL=/mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
export NEMO_SKILLS_TEST_MODEL_TYPE=llama

# generation/evaluation tests
CUDA_VISIBLE_DEVICES='' pytest tests/gpu-tests/test_run_cmd_llm_infer.py -s -x

