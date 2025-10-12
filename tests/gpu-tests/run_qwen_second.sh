# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
set -e


export CUDA_VISIBLE_DEVICES=1
export MKL_SERVICE_FORCE_INTEL=1

export NEMO_SKILLS_TEST_MODEL_TYPE=qwen
# TRTLLM still doesn't support Qwen3 models, using a smaller Qwen2.5 model for context retry tests
export NEMO_SKILLS_TEST_HF_MODEL=Qwen/Qwen2.5-3B-Instruct
# pytest tests/gpu-tests/test_context_retry.py -s -x

# Switch to Qwen3 model for other tests
export NEMO_SKILLS_TEST_HF_MODEL=Qwen/Qwen3-4B
# generation/evaluation tests
pytest tests/gpu-tests/test_generate.py -s -x
