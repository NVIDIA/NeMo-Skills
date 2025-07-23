# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import tqdm

from nemo_skills import utils
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter questions based on token length")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the filtered JSONL file')
    parser.add_argument('--filter_len', type=int, default=3200, help='Maximum token length for filtering')

    args = parser.parse_args()

    print("Filtering questions based on token length...")

    # Load data
    data = utils.jload(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-0528')

    filtered_dataset = []
    for sample in tqdm.tqdm(data, total=len(data)):
        tokenized_input = tokenizer.encode(sample['question'], add_special_tokens=True)
        if len(tokenized_input) > args.filter_len:
            continue

        filtered_dataset.append(sample)

    print("Num samples dropped:", len(data) - len(filtered_dataset))
    utils.jdump(filtered_dataset, args.output_file)

    print(f"Filtered dataset saved to {args.output_file}")