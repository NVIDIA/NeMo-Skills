from os import path
import json
import argparse

from collections import Counter
from utils import extract_judgment


def assign_rm_values(input_file, output_dir):
    counter_fmt = Counter()

    reasoning_instances = set()
    total_instances = 0

    with open(input_file, "r") as f:
        for line in f:
            comp_instance = json.loads(line)
            if comp_instance is None:
                continue
            if "gen_rm_comparison" in comp_instance:
                rm_judgment = comp_instance["gen_rm_comparison"]
            else:
                raise ValueError("No judgment found")

            max_idx = comp_instance["max_idx"]
            judgment = extract_judgment(rm_judgment, max_idx)
            correct_fmt = judgment is not None
            counter_fmt[correct_fmt] += 1
            total_instances += 1

            if not correct_fmt:
                continue
            else:
                if (comp_instance[f"label_{judgment}"] == "Correct"):
                    instance = {
                        "problem": comp_instance["problem"],
                        "solutions": comp_instance["solutions"],
                        "generation": rm_judgment,
                        "max_idx": max_idx,
                        "num_solutions": max_idx + 1
                    }

                    for i in range(max_idx + 1):
                        instance[f"label_{i}"] = comp_instance[f"label_{i}"]
                        instance[f"predicted_answer_{i}"] = comp_instance[f"predicted_answer_{i}"]

                    instance["expected_answer"] = comp_instance["expected_answer"]
                    instance["judgment"] = judgment

                    instance["predicted_answer"] = comp_instance[f"predicted_answer_{judgment}"]
                    reasoning_instances.add(tuple(sorted(list(instance.items()))))


    print("Formatting of judgment: %s" % str(counter_fmt))
    print("# of instances: %d" % len(reasoning_instances))
    print("# of total instances: %d" % total_instances)


    reasoning_instances = list(reasoning_instances)

    output_file = f"{output_dir}/output.jsonl"
    with open(output_file, "w") as fout:
        for instance in reasoning_instances:
            instance = dict(instance)
            fout.write(json.dumps(instance) + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    assign_rm_values(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()