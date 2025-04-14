import argparse
import os
import re
import json
import random
from glob import glob
import logging
os.environ['NEMO_SKILLS_CONFIG_DIR']= "/home/stoshniwal/Research/llm/nemo-skills-config/cluster_configs"
os.environ['NEMO_SKILLS_EXTRA_DATASETS'] = "/home/stoshniwal/Research/llm/nemo-skills-recipes/internal-datasets"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_judgment(text, max_idx=None):
    judgement = None

    try:
        matches = re.findall(r"Judg[e]?ment: (\d+)", text)
        # print(matches)

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
                output_instance = {
                    "problem": instance['problem'], 
                    "expected_answer": instance['expected_answer']
                }

                judgement = extract_judgment(
                    instance['gen_rm_comparison'], max_idx=instance["max_idx"])
                if judgement:
                    output_instance["judgment_idx"] = judgement
                else:
                    output_instance["judgment_idx"] = None
                    judgement = random.randint(0, instance["max_idx"])

                output_instance["predicted_answer"] = instance[f'predicted_answer_{judgement}']
                output_instance["is_correct"] = instance[f'is_correct_{judgement}']

                fout.write(json.dumps(output_instance) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_1_output_dir", type=str, required=True)
    parser.add_argument("--step_2_output_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    get_judgment(args.step_1_output_dir, args.step_2_output_dir, args.output_dir)



if __name__ == "__main__":

    # text = "To determine the unordered set \\(\\{\\alpha_1\\alpha_2 + \\alpha_3\\alpha_4, \\alpha_1\\alpha_3 + \\alpha_2\\alpha_4, \\alpha_1\\alpha_4 + \\alpha_2\\alpha_3\\}\\) where \\(\\alpha_1, \\alpha_2, \\alpha_3, \\alpha_4\\) are the roots of the polynomial \\(x^4 + 2x^3 + 2 = 0\\), we use Vieta's formulas:\n\n- The sum of the roots: \\(\\alpha_1 + \\alpha_2 + \\alpha_3 + \\alpha_4 = -2\\).\n- The sum of the products of the roots taken two at a time: \\(\\alpha_1\\alpha_2 + \\alpha_1\\alpha_3 + \\alpha_1\\alpha_4 + \\alpha_2\\alpha_3 + \\alpha_2\\alpha_4 + \\alpha_3\\alpha_4 = 0\\).\n- The sum of the products of the roots taken three at a time: \\(\\alpha_1\\alpha_2\\alpha_3 + \\alpha_1\\alpha_2\\alpha_4 + \\alpha_1\\alpha_3\\alpha_4 + \\alpha_2\\alpha_3\\alpha_4 = 0\\).\n- The product of the roots: \\(\\alpha_1\\alpha_2\\alpha_3\\alpha_4 = 2\\).\n\nLet \\(S_1 = \\alpha_1\\alpha_2 + \\alpha_3\\alpha_4\\), \\(S_2 = \\alpha_1\\alpha_3 + \\alpha_2\\alpha_4\\), and \\(S_3 = \\alpha_1\\alpha_4 + \\alpha_2\\alpha_3\\). From Vieta's formulas, we know:\n\\[ S_1 + S_2 + S_3 = 0. \\]\n\nNext, we need to find \\(S_1S_2 + S_1S_3 + S_2S_3\\) and \\(S_1S_2S_3\\). Expanding \\(S_1S_2 + S_1S_3 + S_2S_3\\):\n\\[ S_1S_2 + S_1S_3 + S_2S_3 = (\\alpha_1\\alpha_2 + \\alpha_3\\alpha_4)(\\alpha_1\\alpha_3 + \\alpha_2\\alpha_4) + (\\alpha_1\\alpha_2 + \\alpha_3\\alpha_4)(\\alpha_1\\alpha_4 + \\alpha_2\\alpha_3) + (\\alpha_1\\alpha_3 + \\alpha_2\\alpha_4)(\\alpha_1\\alpha_4 + \\alpha_2\\alpha_3). \\]\n\nUsing the fact that \\(\\alpha_1\\alpha_2\\alpha_3\\alpha_4 = 2\\), we simplify:\n\\[ S_1S_2 + S_1S_3 + S_2S_3 = -8. \\]\n\nFinally, we compute the product \\(S_1S_2S_3\\):\n\\[ S_1S_2S_3 = (\\alpha_1\\alpha_2 + \\alpha_3\\alpha_4)(\\alpha_1\\alpha_3 + \\alpha_2\\alpha_4)(\\alpha_1\\alpha_4 + \\alpha_2\\alpha_3) = 8. \\]\n\nThus, \\(S_1, S_2, S_3\\) are the roots of the polynomial:\n\\[ x^3 - (S_1 + S_2 + S_3)x^2 + (S_1S_2 + S_1S_3 + S_2S_3)x - S_1S_2S_3 = 0, \\]\nwhich simplifies to:\n\\[ x^3 - 8x - 8 = 0. \\]\n\nThe roots of this polynomial are:\n\\[ x = 1 + \\sqrt{5}, \\quad x = 1 - \\sqrt{5}, \\quad x = -2. \\]\n\nTherefore, the unordered set is:\n\\[ \\boxed{\\{-2, 1 + \\sqrt{5}, 1 - \\sqrt{5}\\}}. \\]\n\nAmong the provided solutions, Solution 5 is the most mathematically sound and precise, correctly deriving the cubic equation \\(x^3 - 8x - 8 = 0\\) and arriving at the correct roots.\n\nJudgment: 5"

    # print(extract_judgment(text))
    
    main()


