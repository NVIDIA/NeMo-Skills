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

import os
import shlex
import subprocess
import sys

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

mname_from = "meta-llama/Llama-2-7b-hf"
mname_tiny = "/output/tiny-llama"
vocab_keep_items = 32000

config = LlamaConfig.from_pretrained(mname_from)
config.update(
    dict(
        hidden_size=16,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=256,
        num_key_value_heads=4,
        vocab_size=vocab_keep_items,
    )
)
print("new config", config)

# create a tiny random model
tiny_model = LlamaForCausalLM(config)
print(f"num of params {tiny_model.num_parameters()}")

# shrink it more and save
tiny_model.bfloat16()  # half-size
tiny_model.save_pretrained(mname_tiny)

# shrink the tokenizer from 32k to 3k vocab
tokenizer_fast = LlamaTokenizerFast.from_pretrained(mname_from)
tmp_dir = f"/tmp/{mname_from}"
tokenizer_fast.save_pretrained(tmp_dir)
# resize tokenizer.json (vocab.txt will be automatically resized on save_pretrained)
# perl  -0777 -pi -e 's|(2999).*|$1},"merges": []}}|msg' tokenizer.json # 0-indexed, so vocab_keep_items-1!
closing_pat = '},"merges": []}}'
cmd = f"perl -0777 -pi -e 's|({vocab_keep_items-1}).*|$1{closing_pat}|msg' {tmp_dir}/tokenizer.json"
# print(f"Running:\n{cmd}")
result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
# print(result)

# reload with modified tokenizer
tokenizer_fast_tiny = LlamaTokenizerFast.from_pretrained(tmp_dir)
tokenizer_fast_tiny.save_pretrained(mname_tiny)

# test the new model and tokenizer function
model_inputs = tokenizer_fast_tiny("Making tiny model", return_tensors="pt")
gen_tokens = tiny_model.generate(**model_inputs, max_new_tokens=100)
print(tokenizer_fast_tiny.batch_decode(gen_tokens, skip_special_tokens=True))
print("Random output should be expected, but no crashing")

print(f"Model+Tokenizer saved in {mname_tiny}")
