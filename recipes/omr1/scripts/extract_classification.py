import argparse
import json


def filter(input_file, output_file_yes, output_file_no, mode):
    with open(input_file, 'r') as infile, open(output_file_yes, 'w') as outfile_yes, open(
        output_file_no, 'w'
    ) as outfile_no:
        for line in infile:
            data = json.loads(line)
            generation = data.pop("generation")
            data[f"classify_{mode}_gen"] = generation
            if generation.startswith(mode):
                outfile_yes.write(json.dumps(data) + '\n')
            else:
                outfile_no.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Will only keep proof questions based on the expected_answer field.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file_yes", help="Path to the output JSONL file")
    parser.add_argument("output_file_no", help="Path to the output JSONL file")
    parser.add_argument("--mode", choices=['proof', 'yes-or-no', 'mcq', 'valid'], required=True)

    args = parser.parse_args()

    filter(args.input_file, args.output_file_yes, args.output_file_no, args.mode)
