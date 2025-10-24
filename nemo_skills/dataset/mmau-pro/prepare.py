# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import json
import os
import subprocess
import zipfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields


def download_mmau_data(download_dir, hf_token):
    """Download and extract MMAU-Pro data.zip file."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    data_zip_path = download_dir / "data.zip"
    extracted_data_dir = download_dir / "data"
    
    if extracted_data_dir.exists() and any(extracted_data_dir.iterdir()):
        print(f"Data already exists at {extracted_data_dir}")
        return extracted_data_dir
    
    if not data_zip_path.exists():
        print(f"Downloading MMAU-Pro data.zip...")
        cmd = [
            "wget", "--header", f"Authorization: Bearer {hf_token}",
            "https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip",
            "-O", str(data_zip_path)
        ]
        subprocess.run(cmd, check=True)
    
    print(f"Extracting data.zip...")
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    return extracted_data_dir


def format_entry(entry, with_audio=False):
    """Format entry for nemo-skills with OpenAI messages and audio support."""
    choices = entry.get("choices", []) or []
    
    if entry.get('audio_path'):
        if isinstance(entry['audio_path'], list):
            entry['audio_path'] = ['/datasets/mmau-pro/' + path if not path.startswith('/') else path for path in entry['audio_path']]
        else:
            if not entry['audio_path'].startswith('/'):
                entry['audio_path'] = '/datasets/mmau-pro/' + entry['audio_path']
    
    formatted_entry = {
        "expected_answer": entry['answer'],
        **get_mcq_fields(entry["question"], choices),
        **{k: v for k, v in entry.items() if k not in ['answer']}
    }
    
    category = entry.get("category", "")
    
    if category == "open":
        content = entry["question"]
    elif choices and len(choices) > 1:
        options_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        content = f"{entry['question']}\n\n{options_text}"
    else:
        content = entry["question"]
    
    user_message = {"role": "user", "content": content}
    
    if entry.get('audio_path'):
        audio_path = entry['audio_path']
        
        if isinstance(audio_path, list) and audio_path:
            user_message["audios"] = [
                {
                    "path": path,
                    "duration": 10.0
                }
                for path in audio_path
            ]
        elif isinstance(audio_path, str):
            user_message["audio"] = {
                "path": audio_path,
                "duration": 10.0
            }
    
    formatted_entry["messages"] = [user_message]
    return formatted_entry


def main():
    parser = argparse.ArgumentParser(description="Prepare MMAU-Pro dataset for nemo-skills")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--with-audio", action="store_true", help="Download audio files (requires HF_TOKEN)")
    parser.add_argument("--download-dir", help="Directory for audio files (required with --with-audio)")
    args = parser.parse_args()
    
    if args.with_audio:
        if not args.download_dir:
            raise ValueError("--download-dir is required when using --with-audio")
        if not os.environ.get('HF_TOKEN'):
            raise ValueError("HF_TOKEN environment variable required for --with-audio")
        download_mmau_data(args.download_dir, os.environ['HF_TOKEN'])
    
    print(f"Loading {args.split} split...")
    dataset = load_dataset("gamma-lab-umd/MMAU-Pro", trust_remote_code=True)[args.split]
    
    output_dir = Path(__file__).parent
    output_file = output_dir / f"{args.split}.jsonl"
    
    print(f"Processing {len(dataset)} entries...")
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset):
            formatted_entry = format_entry(entry, with_audio=args.with_audio)
            fout.write(json.dumps(formatted_entry) + "\n")
    
    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    main()
