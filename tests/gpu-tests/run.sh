# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
# example: HF_TOKEN=<> ./tests/gpu-tests/run.sh /mnt/datadrive/models/Meta-Llama-3-8B /mnt/datadrive/models/Meta-Llama-3-8B-Instruct
set -e

if [ $# -eq 0 ] ; then
    echo 'Provide llama3-base HF model folder as the first argument'
    exit 1
fi

export NEMO_SKILLS_TEST_HF_MODEL=$1
export NEMO_SKILLS_TEST_OUTPUT=/tmp/nemo_skills_test_output
mkdir -p $NEMO_SKILLS_TEST_OUTPUT

# first running the conversion tests
pytest tests/gpu-tests/test_conversion.py -k test_hf_trtllm_conversion -s
export NEMO_SKILLS_TEST_TRTLLM_MODEL=$NEMO_SKILLS_TEST_OUTPUT/trtllm-model
pytest tests/gpu-tests/test_conversion.py -k test_hf_nemo_conversion -s
export NEMO_SKILLS_TEST_NEMO_MODEL=$NEMO_SKILLS_TEST_OUTPUT/model.nemo
pytest tests/gpu-tests/test_conversion.py -k test_nemo_hf_conversion -s
# using the back-converted model to check that it's reasonable
export NEMO_SKILLS_TEST_HF_MODEL=$NEMO_SKILLS_TEST_OUTPUT/hf-model

export LLAMA3_8B_BASE_TRTLLM=$NEMO_SKILLS_TEST_TRTLLM_MODEL
export LLAMA3_8B_BASE_NEMO=$NEMO_SKILLS_TEST_NEMO_MODEL
export LLAMA3_8B_BASE_HF=$NEMO_SKILLS_TEST_HF_MODEL
export LLAMA3_8B_INSTRUCT_HF=$2

# then running the rest of the tests
pytest tests/gpu-tests/test_generation.py -s

# for sft we are using the tiny random llama model to run much faster
python pipeline/launcher.py \
    --cmd "HF_TOKEN=$HF_TOKEN python /code/tests/gpu-tests/make_tiny_llama.py" \
    --tasks_per_node 1 \
    --job_name make_llama \
    --container igitman/nemo-skills-sft:0.3.0 \
    --mounts $NEMO_SKILLS_TEST_OUTPUT:/output,`pwd`:/code

# converting the model through test
export NEMO_SKILLS_TEST_HF_MODEL=$NEMO_SKILLS_TEST_OUTPUT/tiny-llama
pytest tests/gpu-tests/test_conversion.py -k test_hf_nemo_conversion -s
# untarring model which is required for checkpoint averaging
mkdir -p $NEMO_SKILLS_TEST_OUTPUT/untarred_nemo
tar xvf $NEMO_SKILLS_TEST_OUTPUT/model.nemo -C $NEMO_SKILLS_TEST_OUTPUT/untarred_nemo
export NEMO_SKILLS_TEST_NEMO_MODEL=$NEMO_SKILLS_TEST_OUTPUT/untarred_nemo
# running finetuning
pytest tests/gpu-tests/test_finetuning.py -s