import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def filter_data_by_category(data, category):
    if category == "all":
        return data, list(set(entry['category'] for entry in data))
    return data.filter(lambda x: x['category'] == category), [category]


def get_output_file(data_dir, split, category):
    sanitized_category = category.replace(" ", "_")
    return data_dir / f"{split}_{sanitized_category}.jsonl"


def format_entry(entry):
    return {
        "question": entry['question'],
        "options": "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(entry['options'])),
        "expected_answer": entry['answer'],
        "topic": entry['category'],
    }


def write_category_data(output_file, data, category):
    """Write data of a specific category to a file."""
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            if entry['category'] == category:
                json.dump(format_entry(entry), fout)
                fout.write("\n")


def main(args):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")[args.split]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    filtered_data, categories = filter_data_by_category(dataset, args.category)

    for category in categories:
        output_file = get_output_file(data_dir, args.split, category)
        write_category_data(output_file, filtered_data, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("validation", "test"),
        help="Dataset split to process."
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=(
            "all", "biology", "business", "chemistry", "computer science", 
            "economics", "engineering", "health", "history", "law", "math", 
            "philosophy", "physics", "psychology", "other"
        ),
        help="Category to filter data by."
    )
    args = parser.parse_args()
    main(args)
