import argparse
import json
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")

    args = parser.parse_args()

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile, open(
        args.output_file + "-dropped", 'w'
    ) as outfile_dropped:
        for line in infile:
            data = json.loads(line)
            generation = data.pop("generation")
            data["problem_extraction_gen"] = generation
            if "No problems identified" in generation:  # dropping this post
                outfile_dropped.write(json.dumps(data) + '\n')
                continue

            # Split and clean problems using regex
            problems = re.split(r'Problem\s+\d+\s*:', generation)[1:]

            for problem_idx, problem_text in enumerate(problems):
                cleaned_text = problem_text.strip()
                if not cleaned_text:
                    continue

                new_entry = data.copy()
                new_entry["problem"] = cleaned_text
                new_entry["extracted_problem_idx"] = problem_idx
                outfile.write(json.dumps(new_entry) + '\n')
