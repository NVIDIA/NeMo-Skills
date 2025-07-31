import argparse
import json
import os.path
from pathlib import Path

from datasets import load_dataset
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    # pack the run and compile code into the jsonl file.
    root_dataset_dir = os.path.dirname(Path(__file__))

    with open(os.path.join(root_dataset_dir, "run")) as f:
        run_code = f.read()
    with open(os.path.join(root_dataset_dir, "compile")) as f:
        compile_code = f.read()

    tests_dataset = load_dataset("open-r1/ioi-test-cases", name="2024", split="train")

    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        test_cases[test_case['problem_id']][test_case['test_name']] = {"input": test_case['test_input'], "output": test_case['test_output']}

    ds = load_dataset("open-r1/ioi", split=args.split)
    entries = []
    for x, item in enumerate(ds):
        # remove the examples for the test split, as we do not need to compute evaluation for them.
        if item['score'] != 0 and args.split == 'test':
            selected_test_cases = test_cases[item['id']]
            tests = [test_cases for k, test_cases in selected_test_cases.items() if k in item['test_names']]

            entries.append({
                'id': x,
                'run': run_code,
                'compile': compile_code,
                'name': item['name'],
                'ioi_id': item['id'],
                'subtask': item['subtask'],
                'question': item['problem'],
                'score': item['score'],
                'grader_files': item['grader_files'],
                'tests': tests
            })

    with open(os.path.join(args.output_dir, f'{args.split}.jsonl'), 'w') as f:
        f.write('\n'.join(json.dumps(x) for x in entries))
