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

from transformers import LlamaConfig, LlamaForCausalLM

mname_from = "meta-llama/Meta-Llama-3-8B"
mname_tiny = "/output/tiny-llama"

config = LlamaConfig.from_pretrained(mname_from)
config.update(
    dict(
        hidden_size=16,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=256,
        num_key_value_heads=4,
    )
)
print("new config", config)

# create a tiny random model
tiny_model = LlamaForCausalLM(config)
print(f"num of params {tiny_model.num_parameters()}")

# shrink it more and save
tiny_model.bfloat16()  # half-size
tiny_model.save_pretrained(mname_tiny)
