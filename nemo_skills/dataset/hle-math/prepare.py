import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def format_entry(entry):
    return {
        "id": entry["id"],
        "problem": entry["question"],
        "expected_answer": entry["answer"],
        "answer_type": entry["answer_type"],
        "solution": entry["rationale"],
        "raw_subject": entry["raw_subject"],
        "category": entry["category"],
        "author_name": entry["author_name"],
        "canary": entry["canary"],
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            if entry["category"] != "Math":
                continue
            if entry["image"]:
                continue
            json.dump(format_entry(entry), fout)
            fout.write("\n")


if __name__ == "__main__":
    dataset = load_dataset("cais/hle", split="test")
    columns_to_keep = ['id', 'question', 'answer', 'answer_type', 'rationale', 
                      'raw_subject', 'category', 'author_name', 'canary', 'image']
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"test.jsonl"
    write_data_to_file(output_file, dataset)