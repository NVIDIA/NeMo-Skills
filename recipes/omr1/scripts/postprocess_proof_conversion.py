import argparse
import glob
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", help="Glob pattern for the input JSONL files")
    parser.add_argument("output_file", help="Path to the output JSONL file")

    args = parser.parse_args()

    with open(args.output_file, 'w') as outfile:
        for input_file in glob.glob(args.input_files):
            with open(input_file, 'r') as infile:
                for line in infile:
                    data = json.loads(line)
                    # there should not be any expected answer, but dropping it just in case
                    data["expected_answer"] = None
                    data["original_problem"] = data.pop("problem")
                    data["problem"] = data.pop("generation")
                    outfile.write(json.dumps(data) + '\n')
