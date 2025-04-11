import argparse
import json

from nemo_skills.code_execution.math_grader import extract_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")

    args = parser.parse_args()

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            generation = data.pop("generation")
            data["answer_extraction_gen"] = generation
            if "Output:" in generation:
                generation = generation.split("Output:")[1].strip()
            if "Answer not found" in generation or "Answer:" not in generation:
                data["expected_answer"] = None
                outfile.write(json.dumps(data) + '\n')
                continue
            generation = generation.split("Answer:")[1].strip()
            # sometimes LLM uses math-problem format..
            if "\\boxed{" in generation:
                generation = extract_answer(generation)
            data["expected_answer"] = generation
            outfile.write(json.dumps(data) + '\n')
