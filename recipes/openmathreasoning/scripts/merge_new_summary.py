import json
import re
import glob
import argparse

from typing import List, Dict, Optional
from os import path
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.code_execution.math_grader import extract_answer



def read_jsonl_file(file_path: str, key: Optional[str] = None) -> List[Dict]:
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            if key is not None:
                instances.append(instance[key])
            else:
                instances.append(instance)
    
    return instances


def is_valid_summary(reasoning_instance: Dict, summary_instance: Dict) -> bool:
    """Identify if the summary is valid for the reasoning solution"""

    # If both the reasoning solution and the summary are judged correct, then the summary is valid
    if is_correct_judgement(reasoning_instance["judgement"]) and is_correct_judgement(summary_instance["judgement"]):
        return True

    # Otherwise check for the surface form to ensure that the summary has the same answer, even if incorrect, as the reasoning solution
    return (reasoning_instance["predicted_answer"] == summary_instance["predicted_answer"])


def select_best_summary(valid_summaries):
    """Select the best summary from the list of valid summaries. 
    Currently we just select the longest valid summary in terms of characters."""

    return max(valid_summaries, key=lambda x: len(x["generation"]))


def trim_reasoning_generation(reasoning_generation, start_tag, end_tag):    
    """Trim the thinking part of the original reasoning generation till the step with the rightmost boxed entry"""
    
    # Find the start and end tags. If either is not found, return None
    start_tag_position = reasoning_generation.find(start_tag)
    if start_tag_position == -1:
        return None, None

    end_tag_position = reasoning_generation.find(end_tag)
    if end_tag_position == -1:
        return None, None

    reasoning_trace = reasoning_generation[:end_tag_position + len(end_tag)]

    # Extract the answer from the reasoning trace by searching for boxed entries
    answer_from_reasoning_trace = extract_answer(reasoning_trace)
    
    # If the answer is found, trim the reasoning trace to the step with the rightmost boxed entry
    if answer_from_reasoning_trace:
        answer_expression = r'\\boxed\{"[ ]*' + re.escape(answer_from_reasoning_trace) + r'[ ]*\}'
        matches = list(re.finditer(answer_expression, reasoning_trace))
    
        # Return the rightmost match if any
        if matches:
            rightmost_match = matches[-1]
            # Remove steps after the rightmost match
            reasoning_trace = (
                reasoning_trace[:rightmost_match.end()] + 
                reasoning_trace[rightmost_match.end():].split("\n\n")[0]
            )

            # If the end tag is not present, add it
            if end_tag not in reasoning_trace:
                reasoning_trace += end_tag
                
    return reasoning_trace


def format_reasoning_trace_with_summary(reasoning_file, summary_dir, start_tag, end_tag):
    """Format the reasoning trace with the best summary from the summary directory"""
    # Read the reasoning instances
    reasoning_instances = read_jsonl_file(reasoning_file)

    # If the summary directory does not exist, return an empty list and the counts
    if not path.exists(summary_dir):
        return [], 0, 0, len(reasoning_instances), 0

    # We have multiple summaries for the same reasoning trace
    list_of_summary_instances = [read_jsonl_file(summary_file) for summary_file in glob.glob(path.join(summary_dir, "*.jsonl"))]

    formatted_instances = []

    # Ensure that the number of summaries is the same as the number of reasoning instances
    list_of_summary_instances = [summary_instances for summary_instances in list_of_summary_instances if len(reasoning_instances)==len(summary_instances)]

    # If there are no valid summaries, return an empty list and the counts
    if len(list_of_summary_instances) == 0:
        invalid_summary_count += len(reasoning_instances)
        return [], 0, 0, len(reasoning_instances), 0

    all_summaries = list(zip(*list_of_summary_instances))
    for (reasoning_instance, summaries_for_reasoning_instance) in zip(reasoning_instances, all_summaries):
        # Step 1 - Trim the reasoning generation
        trimmed_reasoning_trace = trim_reasoning_generation(reasoning_instance["generation"], start_tag, end_tag)

        # If the reasoning generation is not trimmed, skip this instance
        if trimmed_reasoning_trace is None:
            continue
        
        valid_summaries = [summary_instance for summary_instance in summaries_for_reasoning_instance 
                        if is_valid_summary(reasoning_instance, summary_instance)]

        if len(valid_summaries) == 0:
            continue  # Can't format this instance with new summary. Skip it.
        else:
            # Select the best summary
            best_summary = select_best_summary(valid_summaries)                
            # Combine the trimmed reasoning trace with the best summary
            combined_generation = trimmed_reasoning_trace + best_summary["generation"]
            # Update the reasoning instance
            reasoning_instance["generation"] = combined_generation
            # Add the instance to the list of formatted instances
            formatted_instances.append(reasoning_instance)

    return formatted_instances



def main():
    parser = argparse.ArgumentParser(description="Merge new summary into the reasoning trace")
    parser.add_argument("--reasoning_file", type=str, required=True, help="Path to the reasoning file")
    parser.add_argument("--summary_dir", type=str, required=True, help="Path to the summary directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--start_tag", type=str, required=True, help="Start tag")
    parser.add_argument("--end_tag", type=str, required=True, help="End tag")
    args = parser.parse_args()

    formatted_instances = format_reasoning_trace_with_summary(args.reasoning_file, args.summary_dir, args.start_tag, args.end_tag)

    with open(args.output_file, "w") as f:
        for instance in formatted_instances:
            f.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    main()