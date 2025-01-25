# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
set -e

export NEMO_SKILLS_TEST_MODEL_TYPE=qwen_orm

docker run --rm \
    -e HF_TOKEN=$HF_TOKEN \
    -v /tmp:/tmp \
    -v `pwd`:/nemo_run/code \
    igitman/nemo-skills-nemo:0.5.0 \
    python /nemo_run/code/tests/gpu-tests/make_tiny_llm.py --model_type $NEMO_SKILLS_TEST_MODEL_TYPE;

export NEMO_SKILLS_TEST_HF_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/tiny-model-hf
pytest tests/gpu-tests/test_reward.py -s -x
