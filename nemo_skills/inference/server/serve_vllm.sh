#!/bin/bash

# Usage: ./serve_vllm.sh {model_path} {model_name} <|Optional|> {seed} {server_type} {server_port}
# Server type can be "openai" (default) or "vllm"

# Example : bash serve_vllm.sh "codellama/CodeLlama-7b-Instruct-hf" "codellama-7b-instruct-hf" 0 "openai" 5000
# Example (file): bash serve_vllm.sh "/cache/huggingface/transformers/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/6114dd1e16f69e0765ccbd7a64d33d04b265fbd2/" "codellama/CodeLlama-7b-Instruct-hf" 0 "openai" 5000

# Example (Mistral): bash serve_vllm.sh "mistralai/Mistral-7B-Instruct-v0.1" "Mistral-7B-Instruct-v0.1" 0 "openai" 5000
# Example (Mixtral): bash serve_vllm.sh "mistralai/Mixtral-8x7B-Instruct-v0.1" "Mixtral-8x7B-Instruct-v0.1" 0 "openai" 5000
# Example (Mixtral AWQ): bash serve_vllm.sh "casperhansen/mixtral-instruct-awq" "casperhansen/mixtral-instruct-awq" 0 "openai" 5000 awq

# Example (Llama 3 70B AWQ): bash serve_vllm.sh "casperhansen/llama-3-70b-instruct-awq" "casperhansen/llama-3-70b-instruct-awq" 0 "openai" 5000 awq

# Example (WizardCoder-Python-7B): bash serve_vllm.sh "WizardLM/WizardCoder-Python-7B-V1.0" "WizardCoder-Python-7B-V1.0" 0 "openai" 5000
# Example (Deepseek-coder-6.7b): bash serve_vllm.sh "deepseek-ai/deepseek-coder-6.7b-instruct" "deepseek-coder-6.7b-instruct" 0 "openai" 5000

# Cache models locally
# Sometimes models may not cache properly. You may need to use conda env to cache the model manually then.
# export TRANSFORMERS_CACHE="~/.cache/huggingface/transformers"
# python
# import transformers
# model = transformers.AutoModelForCausalLM.from_pretrained("<MODEL NAME>")

# Model Names
# codellama/CodeLlama-7b-Instruct-hf
# HuggingFaceH4/zephyr-7b-alpha

# Getting the local container Host address (IP)
# $ docker ps
# Get the id of the container
# $ docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id>

# Example call via curl
<<exmpl

curl http://<ContainerIPAddress>:5000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "codellama/CodeLlama-7b-Instruct-hf",  # needed for openai only
  "prompt": "[INST] Write a python function to calculate the first n fibonacci series. [/INST]",
  "temperature": 0, "max_tokens": 100
  }'

exmpl

# Get model path
MODEL_PATH=${1:?"Missing model path"}
MODEL_NAME=${2:?"Missing model name"}  # Only used for OpenAI Server

echo "Deploying model ${MODEL_NAME} from ${MODEL_PATH}"

SEED=${3:-1}
SERVER_TYPE=${4:-"openai"}
SERVER_PORT=${5:-5000}
QUANTIZATION=${6:-""}

# Deploy model with all gpus on local machine
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# Check transformers cache and set download-dir if necessary
if [ -n "$QUANTIZATION" ]
then
      QUANTIZATION="--quantization ${QUANTIZATION}"
else
      QUANTIZATION=""
fi

# Select server
# Start OpenAI Server
echo "Starting OpenAI Server"
printf "\n\n"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --seed=${SEED} \
  --host="0.0.0.0" \
  --port=${SERVER_PORT} \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size=${NUM_GPUS} \
  --max-num-seqs=1024 \
  --enforce-eager \
  --disable-log-requests $QUANTIZATION
