import argparse
import json
import os
from datetime import datetime
from datasets import load_dataset
from dateutil.relativedelta import relativedelta


class PromptConstants:
    # reference: https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/reasoning/livecode_bench_cot/lcb_runner_cq/prompts/code_generation.py#L31
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def parse_data(release_version='release_latest'):
    data = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release_version,
        trust_remote_code=True
    )
    # data has the following fields
    # question_title: str
    # question_content: str
    # platform: Platform
    # question_id: str
    # contest_id: str
    # contest_date: datetime
    # starter_code: str
    # difficulty: Difficulty
    # public_test_cases: list[Test]
    # private_test_cases: list[Test]
    # metadata: dict
    return data


def get_first_last_day(year_month_str):
    try:
        date_obj = datetime.strptime(year_month_str, "%Y-%m")
        first_day = date_obj.date().replace(day=1)
        last_day = (date_obj + relativedelta(months=1, days=-1)).date()
        return first_day, last_day
    except ValueError:
        raise ValueError("Invalid date format. Please use '%Y-%m'.")


def parse_month_range(start_date, end_date):
    try:
        start_date, _ = get_first_last_day(start_date)
        _, end_date = get_first_last_day(end_date)
        return start_date, end_date
    except ValueError as e:
        raise ValueError(str(e))


def clean_data(dataset):
    def map_fn(data):
        question = data["question_content"] + "\n\n"
        if data["starter_code"]:
            question += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            question += f"```python\n{data['starter_code']}\n```\n\n"
        else:
            question += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
            question += f"```python\n# YOUR CODE HERE\n```\n\n"

        data["task_id"] = data["question_id"]
        data['question'] = question.replace('    ', '\t')
        return data

    remove_columns = [
        'question_title', 'contest_id',
        'public_test_cases', 'private_test_cases', 'metadata',
        'question_content', 'platform', 'question_id', 'starter_code'
    ]
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    return dataset


if __name__ == '__main__':
    # Write an argparse to a json file, read it in and parse it
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--release_version', type=str, default='v5')
    parser.add_argument('--start_date', type=str, default='2024-08', help="End date in YYYY-MM format")
    parser.add_argument('--end_date', type=str, default='2025-02', help="End date in YYYY-MM format")

    args = parser.parse_args()

    start_date, end_date = parse_month_range(args.start_date, args.end_date)
    start_yymm = start_date.strftime("%y%m")
    end_yymm = end_date.strftime("%y%m")
    output_file_path = os.path.join(args.output_dir, f"test_{args.release_version}_{start_yymm}_{end_yymm}.jsonl")

    assert args.release_version in ["v1", "v2", "v3", "v4", "v5", "v6"]

    data = parse_data(release_version=f"release_{args.release_version}")
    data = clean_data(data)
    print("Len of data: ", len(data))

    print("Writing to file...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_file_path, 'w') as f:
        for problem in data:
            input_date = datetime.strptime(problem['contest_date'], '%Y-%m-%dT%H:%M:%S').date()
            if start_date <= input_date <= end_date:
                json.dump({
                    "task_id": problem["task_id"],
                    "question": problem["question"],
                    "difficulty": problem["difficulty"]
                }, f)
                f.write('\n')

    # test_v5_2408_2502.jsonl: 279 samples
    # test_v5_2410_2502.jsonl: 166 samples
    # test_v5_2410_2504.jsonl: 166 samples
    # test_v6_2408_2502.jsonl: 374 samples
    # test_v6_2408_2502.jsonl: 261 samples
    # test_v6_2410_2504.jsonl: 341 samples
