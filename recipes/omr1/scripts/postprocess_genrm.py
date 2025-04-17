import argparse
import json
import logging
import os
import random
import re
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_judgment(text, max_idx=None):
    judgement = None

    try:
        matches = re.findall(r"Judg[e]?ment: (\d+)", text)

        if matches:
            number = matches[-1]
            judgement = int(number)
            if max_idx is not None and judgement > max_idx:
                judgement = None
        else:
            judgement = None

    except:
        judgement = None

    if judgement is not None and max_idx is not None:
        if judgement > max_idx:
            judgement = None

    return judgement


def get_judgment(step_1_output_dir, step_2_output_dir, output_dir):
    num_random_seeds = len(glob(os.path.join(step_2_output_dir, "output-rs*.jsonl")))
    logger.info(f"Number of random seeds: {num_random_seeds}")

    single_answer_instances_file = os.path.join(step_1_output_dir, "single_answer_instances.jsonl")
    single_answer_instances = [json.loads(line) for line in open(single_answer_instances_file, "r")]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for random_seed in range(num_random_seeds):
        input_file = os.path.join(step_2_output_dir, f'output-rs{random_seed}.jsonl')
        output_file = os.path.join(output_dir, f'output-rs{random_seed}.jsonl')

        with open(input_file, 'r') as f, open(output_file, 'w') as fout:
            for single_answer_instance in single_answer_instances:
                fout.write(json.dumps(single_answer_instance) + '\n')

            for line in f:
                instance = json.loads(line)
                output_instance = {"problem": instance['problem'], "expected_answer": instance['expected_answer']}

                judgement = extract_judgment(instance['gen_rm_comparison'], max_idx=instance["max_idx"])
                if judgement:
                    output_instance["judgment_idx"] = judgement
                else:
                    output_instance["judgment_idx"] = None
                    judgement = random.randint(0, instance["max_idx"])

                output_instance["predicted_answer"] = instance[f'predicted_answer_{judgement}']
                output_instance["is_correct"] = instance[f'is_correct_{judgement}']
                output_instance["subset_for_metrics"] = instance["subset_for_metrics"]
                fout.write(json.dumps(output_instance) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_1_output_dir", type=str, required=True)
    parser.add_argument("--step_2_output_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    get_judgment(args.step_1_output_dir, args.step_2_output_dir, args.output_dir)


if __name__ == "__main__":
    main()
