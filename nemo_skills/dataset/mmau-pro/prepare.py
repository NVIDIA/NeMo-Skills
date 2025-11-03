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

from nemo_skills.dataset.utils import generate_subgroup_init_files, get_mcq_fields


def _get_mmau_pro_subgroup_configs():
    """Define the __init__.py configurations for each MMAU-Pro subgroup."""
    return {
        "closed_form": """# Closed-form questions evaluated with NV Embed similarity matching
METRICS_TYPE = "speechlm"

# No EVAL_ARGS - evaluation happens only during judge phase with NVEmbed
GENERATION_ARGS = "++prompt_format=openai"

# Split into 10 chunks for parallel processing of large dataset
NUM_CHUNKS = 10

# Use NVEmbed judge for closed-form evaluation
JUDGE_PIPELINE_ARGS = {
    "judge_type": "nvembed",
}""",
        "open_ended": """# Open-ended questions evaluated with LLM judge (Qwen)
METRICS_TYPE = "speechlm"
GENERATION_ARGS = "++prompt_format=openai"

# Single chunk (small dataset)
NUM_CHUNKS = 1

# Judge configuration for open-ended evaluation using NVIDIA API
JUDGE_PIPELINE_ARGS = {
    "model": "qwen/qwen2.5-7b-instruct",
    "server_type": "openai",
    "server_address": "https://integrate.api.nvidia.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/speechlm ++generation_key=judgement"
""",
        "instruction_following": """# Instruction following questions evaluated with AIF format
METRICS_TYPE = "speechlm"
EVAL_ARGS = "++eval_type=mmau-pro"
GENERATION_ARGS = "++prompt_format=openai"

# Single chunk (small dataset)
NUM_CHUNKS = 1""",
    }


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
        print("Downloading MMAU-Pro data.zip...")
        cmd = [
            "wget",
            "--header",
            f"Authorization: Bearer {hf_token}",
            "https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip",
            "-O",
            str(data_zip_path),
        ]
        subprocess.run(cmd, check=True)

    print("Extracting data.zip...")
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    return extracted_data_dir


def format_entry(entry, with_audio=False):
    """Format entry for nemo-skills with OpenAI messages and audio support."""
    choices = entry.get("choices", []) or []

    formatted_entry = {
        "expected_answer": entry["answer"],
        **get_mcq_fields(entry["question"], choices),
        **{k: v for k, v in entry.items() if k not in ["answer"]},
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

    if entry.get("audio_path"):
        audio_path = entry["audio_path"]

        if isinstance(audio_path, list) and audio_path:
            user_message["audios"] = [{"path": path, "duration": 10.0} for path in audio_path]
        elif isinstance(audio_path, str):
            user_message["audio"] = {"path": audio_path, "duration": 10.0}

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
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("HF_TOKEN environment variable required for --with-audio")
        download_mmau_data(args.download_dir, os.environ["HF_TOKEN"])

    print(f"Loading {args.split} split...")
    dataset = load_dataset("gamma-lab-umd/MMAU-Pro", trust_remote_code=True)[args.split]

    output_dir = Path(__file__).parent

    # Separate files for each evaluation category
    category_files = {
        "closed_form": output_dir / "closed_form" / f"{args.split}.jsonl",
        "open": output_dir / "open_ended" / f"{args.split}.jsonl",
        "instruction following": output_dir / "instruction_following" / f"{args.split}.jsonl",
    }

    for category_file in category_files.values():
        category_file.parent.mkdir(parents=True, exist_ok=True)
    
    generate_subgroup_init_files(output_dir, _get_mmau_pro_subgroup_configs())

    category_file_handles = {
        category: open(file_path, "w", encoding="utf-8") for category, file_path in category_files.items()
    }

    print(f"Processing {len(dataset)} entries into separate category files...")
    category_counts = {category: 0 for category in category_files.keys()}

    try:
        for entry in tqdm(dataset):
            formatted_entry = format_entry(entry, with_audio=args.with_audio)
            category = entry.get("category", "closed_form")

            # Map category to file
            if category == "instruction following":
                target_category = "instruction following"
            elif category == "open":
                target_category = "open"
            else:
                # All other categories (closed-form, multiple choice) go to closed_form
                target_category = "closed_form"

            category_file_handles[target_category].write(json.dumps(formatted_entry) + "\n")
            category_counts[target_category] += 1
    finally:
        for fh in category_file_handles.values():
            fh.close()

    print("Dataset split into categories:")
    for category, file_path in category_files.items():
        print(f"  - {category}: {category_counts[category]} entries -> {file_path}")


if __name__ == "__main__":
    main()
