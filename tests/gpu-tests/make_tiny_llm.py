# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://huggingface.co/stas/tiny-random-llama-2/blob/main/make_tiny_model.py

import argparse

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Create a tiny model for testing.")
parser.add_argument("--model_type", type=str, required=True, choices=("qwen", "llama", "qwen_orm"))
args = parser.parse_args()

if args.model_type == 'qwen':
    model_name = "Qwen/Qwen2.5-1.5B"
    output_dir = "/tmp/nemo-skills-tests/qwen/tiny-model-hf"
    hidden_dim = 64
    head_dim = 2
    max_position_embeddings = 256
    num_attention_heads = 8
elif args.model_type == 'qwen_orm':
    # vLLM requires a minimum head dimension size of 32, so we use a larger value here
    model_name = "Qwen/Qwen2.5-Math-RM-72B"
    output_dir = "/tmp/nemo-skills-tests/qwen_orm/tiny-model-hf"
    hidden_dim = 256
    head_dim = 32
    num_attention_heads = 8
    max_position_embeddings = 2048
else:
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "/tmp/nemo-skills-tests/llama/tiny-model-hf"
    hidden_dim = 2048
    head_dim = 64
    max_position_embeddings = 131072
    num_attention_heads = 32


config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.update(
    dict(
        hidden_size=hidden_dim,
        head_dim=head_dim,
        intermediate_size=hidden_dim,
        num_hidden_layers=2,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
)
print("new config", config)

if args.model_type == 'qwen_orm':
    tiny_model = AutoModel.from_config(config, trust_remote_code=True)
else:
    # create a tiny random model
    tiny_model = AutoModelForCausalLM.from_config(config)

print(f"# of params: {tiny_model.num_parameters() / 1_000_000:.1f}M")

# shrink it more and save
tiny_model.bfloat16()  # half-size
tiny_model.save_pretrained(output_dir)

hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_tokenizer.save_pretrained(output_dir)
