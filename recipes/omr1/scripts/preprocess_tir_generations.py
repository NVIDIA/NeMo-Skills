import argparse
import glob
import json
import os
from collections import Counter

# TODO: 
# move unnecessary preprocessing to the prepare sft data, add replace code tag

def validate_code_execution(text, code_begin="```python", code_end="```"):
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        # Look for the start of code execution
        if lines[i] == code_begin:
            # Find the end of code execution
            code_end_idx = -1
            for j in range(i+1, len(lines)):
                if lines[j] == code_end:
                    code_end_idx = j
                    break
            
            # If no end marker found, pattern is invalid
            if code_end_idx == -1:
                return False
            
            # Check for output marker after code execution
            if code_end_idx + 1 >= len(lines) or lines[code_end_idx + 1] != "```output":
                return False
            
            # Find the end of output
            output_end_idx = -1
            for j in range(code_end_idx + 2, len(lines)):
                if lines[j] == "```":
                    output_end_idx = j
                    break
            
            # If no end marker for output found, pattern is invalid
            if output_end_idx == -1:
                return False
            
            # Move index past this complete code execution + output block
            i = output_end_idx + 1
        else:
            # Move to next line
            i += 1
    
    return True


def filter_code_solution(sample, args):
    required_keys = ["predicted_answer", "generation", "problem"]
    for key in required_keys:
        if key not in sample:
            return "Key not found: " + key
    
    # Make some initial filtering to speed up the next llm judgement stage
    if args.code_begin not in sample["generation"]:
        return "No code blocks found"
    if not validate_code_execution(sample["generation"], args.code_begin.strip("\n"), args.code_end.strip("\n")):
        return "Incomplete code execution found"
    if "judgement" in sample and "judgement: no" in sample["judgement"].lower():
        return "Incorrect final answer"
    # if sample["predicted_answer"].lower().startswith("yes") or sample["predicted_answer"].lower().startswith("no"):
    #     return "Predicted answer: Yes/No"
    # if sample["generation"].find("\\boxed{") != -1 and sample["generation"].find("\\boxed{") < sample["generation"].find(args.code_begin):
    #     return "Boxed before code"
    # if sample["generation"].find(sample["predicted_answer"]) != -1 and sample["generation"].find(sample["predicted_answer"]) < sample["generation"].find(args.code_begin):
    #     return "Predicted answer before code" # TODO: move this to prepare sft data for transparency
    
    # generation = cut_final_answer_part(sample["generation"])
    # if generation is None:
    #     return "Final answer not found"
    
    # sample["generation"] = generation

    return "Accepted"

def preprocess_code_judge(args):
    cnt = Counter()
    with open(args.output_file, "w") as fout:
        for input_file in glob.glob(args.input_files):
            with open(input_file, "r") as fin:
                for idx, line in enumerate(fin):
                    sample = json.loads(line)
                    filt_reason = filter_code_solution(sample, args)
                    cnt[filt_reason] += 1
                    cnt["Total"] += 1
                    if filt_reason is "Accepted":
                        sample["original_index"] = idx
                        fout.write(json.dumps(sample) + "\n")
    
    print("Filtered samples:")
    for key, value in cnt.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess code judge data")
    parser.add_argument("--input_files", type=str, required=True, help="Input file, could be a pattern like output*.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--code_begin", type=str, default="```python", help="Start of code execution block tag")
    parser.add_argument("--code_end", type=str, default="```", help="End of code execution block tag")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    preprocess_code_judge(args)