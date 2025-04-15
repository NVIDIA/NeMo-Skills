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


import json
import multiprocessing as mp

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_TOKENS = 24000


def clean_quoted_text(text):
    # Simple pattern to match anything between [quote] and [/quote]
    # including nested quotes
    while '[quote' in text and '[/quote]' in text:
        # Find the last occurring [quote] (handles nesting)
        start = text.rfind('[quote')
        if start == -1:
            break

        # Find the first [/quote] after that position
        end = text.find('[/quote]', start)
        if end == -1:
            break

        # Remove the quoted text including the tags
        text = text[:start] + text[end + 8 :]

    return text


def process_element(elem, tokenizer, max_tokens):
    data = {}
    data['forum_post'] = elem['original_question']
    data["forum_discussions"] = ""
    current_text = ""
    current_tokens = 0

    for idx, post in enumerate(elem["original_answers"], 1):
        cleaned_post = clean_quoted_text(post)
        new_post = f"Post {idx}:\n{cleaned_post.strip()}\n\n"

        new_tokens = len(tokenizer.encode(new_post))
        if current_tokens + new_tokens > max_tokens:
            break

        current_tokens += new_tokens
        current_text += new_post

    data["forum_discussions"] = current_text.strip()
    return data


def init_worker():
    # This prevents the tokenizer from being re-downloaded for each worker
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")


def process_element_wrapper(elem):
    # Use the global tokenizer initialized in each worker
    return process_element(elem, worker_tokenizer, MAX_TOKENS)


if __name__ == "__main__":
    dataset = load_dataset("DeepStudentLlama/AoPS-Instruct", "2024_not_decontaminated")

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")

    # Create a pool of workers
    with mp.Pool(num_workers, initializer=init_worker) as pool:
        # Process the dataset in parallel with a progress bar
        processed_data = list(
            tqdm(
                pool.imap(process_element_wrapper, dataset['train']),
                total=len(dataset['train']),
                desc="Processing dataset",
            )
        )

    print(f"Processed {len(processed_data)} elements")
    with open("raw_aops_data.jsonl", "w") as fout:
        for item in processed_data:
            fout.write(json.dumps(item) + "\n")
