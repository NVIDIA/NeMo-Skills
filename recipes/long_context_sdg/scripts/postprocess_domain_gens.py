import json
import argparse
import ast

def extract_generation_fields(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            generation = data.get("generation", "").strip()
            if generation:
                try:
                    # Safely parse the generation string to a dict
                    gen_dict = ast.literal_eval(generation.split("<|channel|>final<|message|>")[-1].strip())
                    data["domain"] = gen_dict.get("domain")
                    data["subtopics"] = gen_dict.get("subtopics")
                except Exception as e:
                    print(f"Warning: Could not parse generation for line: {line}\nError: {e}")
                    data["domain"] = None
                    data["subtopics"] = None
            else:
                data["domain"] = None
                data["subtopics"] = None
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract domain and subtopics from generation field in JSONL.")
    parser.add_argument("input_file", type=str, help="Input JSONL file path")
    parser.add_argument("output_file", type=str, help="Output JSONL file path")
    args = parser.parse_args()

    extract_generation_fields(args.input_file, args.output_file)
