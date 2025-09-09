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
import argparse
from cgitb import text
import json
from re import A
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

"""
Usage
# default. setup is aalcr (all).
python prepare.py

# prepare subset aalcr_100k.
python prepare.py --max_context_window 100000 --setup aalcr_100k

or 
ns prepare_data \
    --data_dir=/workspace/ns-data \
    --cluster=fei-ord \
    aalcr --txt_file_folder=/workspace/do_not_share_data/lcr
"""

def construct_prompt(docs, question):
    documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
    prompt = """BEGIN INPUT DOCUMENTS

    {documents_text}

    END INPUT DOCUMENTS

    Answer the following question using the input documents provided above.

    START QUESTION

    {question}

    END QUESTION
    """.format(documents_text=documents_text, question=question).strip()
    return prompt


def count_n_tokens(prompt: str, tokenizer_name: str) -> int:
    """
    count tokens with tokenizer, default is cl100k_base. You can use other tokenizers with AutoTokenizer
    """
    enc = tiktoken.get_encoding(tokenizer_name)
    return len(enc.encode(prompt))


def write_data_to_file(output_file, data, txt_file_folder, max_context_window, tokenizer_name):
    
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, entry in tqdm(enumerate(data), desc=f"Writing {output_file.name}"):
            
            entry['index'] = entry.pop('question_id')
            
            document_set_id = entry.pop('document_set_id')
            document_category = entry['document_category']
            data_source_filenames = entry.pop('data_source_filenames').split(';')
            
            # Collect documents
            documents = []
            for data_source_filename in data_source_filenames:
                try:
                    with open(f"{txt_file_folder}/{document_category}/{document_set_id}/{data_source_filename}", "rt", encoding="utf-8") as fin:
                        document = fin.read()
                        documents.append(document)
                except FileNotFoundError:
                    print(f"File {txt_file_folder}/{document_category}/{document_set_id}/{data_source_filename} is missing")
                    continue
            # Use construct_prompt to format the question with documents
            question_text = entry.pop('question')
            question = construct_prompt(documents, question_text)
            
            # find n_tokens with tokenizer_name
            n_tokens = count_n_tokens(question, tokenizer_name)
            if max_context_window is not None:
                if n_tokens > max_context_window:
                    print(f"Skipping {idx} because it has {n_tokens} tokens")
                    continue
    
            entry[f'n_tokens_{tokenizer_name}'] = n_tokens
            entry['question'] = question
            entry['expected_answer'] = entry.pop('answer')
            entry['expected_judgement'] = 'correct' # for judgement metric
            # remove unused columns
            entry.pop("data_source_urls")

            json.dump(entry, fout)
            fout.write("\n")
            
def get_aalcr_data(txt_file_folder, max_context_window, setup, tokenizer_name):
    dataset = load_dataset("ArtificialAnalysis/AA-LCR")['train'] # testset but named as train
    
    data_dir = Path(__file__).absolute().parent

    if txt_file_folder == "lcr":
        txt_file_folder =  Path(__file__).absolute().parent / "lcr"
    
    if not Path(txt_file_folder).exists():
        raise ValueError("txt_file_folder is required. Please consult with AA or process using data_source_urls")

    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, dataset, txt_file_folder, max_context_window, tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MRCR dataset.")
    parser.add_argument(
        "--max_context_window",
        type=int,
        default=None,
        help="Maximum context window size.",
    )
    parser.add_argument(
        "--txt_file_folder",
        type=str,
        default="lcr", #lcr-document-sets
        help="txt file folder to process",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="aalcr",
        help="setup name. e.g. aalcr or aalcr_64k",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="cl100k_base",
        help="tokenizer name",
    )
    
    args = parser.parse_args()

    print(f"Preparing AA-LCR dataset with sadditional arguments: {args}")
    get_aalcr_data(args.txt_file_folder, args.max_context_window, args.setup, args.tokenizer_name)
    print(f"AA-LCR dataset preparation with setup {args.setup} completed. Use --split=${args.setup} to evaluate!")