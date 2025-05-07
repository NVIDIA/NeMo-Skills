# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Utils for GenSelect pipeline"""


from scipy import stats
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
import random
import re


def _format_instance(instance, max_solutions=16):
    solutions = []
    num_solutions = max_solutions
    for i in range(max_solutions):
        if f"solution_{i}" in instance:
            solutions.append(instance[f"solution_{i}"])
        else:
            num_solutions = i
            break

    max_idx = num_solutions - 1

    consolidated_solutions = ""
    for solution in solutions:
        consolidated_solutions += f"Solution {solutions.index(solution)}:\n{solution}\n\n"

    consolidated_solutions = consolidated_solutions.rstrip("\n")
    new_instance = {
        "problem": instance["problem"], 
        "solutions": consolidated_solutions, 
        "max_idx": max_idx, 
        "num_solutions": num_solutions,
        "expected_answer": instance["expected_answer"]
    }

    for idx in range(num_solutions):
        new_instance[f"predicted_answer_{idx}"] = instance[f"predicted_answer_{idx}"]
        new_instance[f"label_{idx}"] = instance[f"label_{idx}"]
    
    return new_instance


def _generate_random_count(min_val=2, max_val=16, peak=8):
    # Calculate parameters
    if max_val == min_val:
        return min_val
    mean = peak
    std = (max_val - min_val) / 6  # This ensures most values fall within range
    
    # Generate truncated normal distribution
    a = (min_val - mean) / std
    b = (max_val - mean) / std

    # Draw one sample from the truncated normal distribution
    sample = stats.truncnorm(a, b, loc=mean, scale=std).rvs(1)[0]
    # Round to the nearest integer
    return int(sample)


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


def extract_summary(reasoning_solution, just_true_summary=False):
    tag = "</think>"
    think_tag_position = reasoning_solution.rfind(tag)
    if think_tag_position != -1:
        summary = reasoning_solution[think_tag_position + len(tag):]
        if summary.count("\\boxed") == 1:
            if len(summary) > 3000:
                summary = summary[-3000:]
                if summary.count("\n\n") > 1:
                    summary = "\n\n".join(summary.split("\n\n")[1:])

            return summary
        else:
            return None

    if just_true_summary:
        # No true summary over here
        return None
    else:
        # Try our best to give a solution which resembles a summary
        reasoning_solution = reasoning_solution.replace("<think>", " ")
        reasoning_solution = reasoning_solution.strip()
        return reasoning_solution
    

def segregate_instances(all_instances):
    """
    Segregate solutions into correct and incorrect based on the judgement. 
    Also remove incorrect solutions without any predicted answer.
    """
    correct_solutions = []
    incorrect_solutions = []
    for instance in all_instances:
        summary = extract_summary(instance["generation"], just_true_summary=True)
        if summary:
            instance["generation"] = summary
            if is_correct_judgement(instance["judgement"]):
                correct_solutions.append(instance)
            else:
                if instance["predicted_answer"] is not None:
                    incorrect_solutions.append(instance)
    return correct_solutions, incorrect_solutions


def create_comparison_instance(correct_solutions, incorrect_solutions, max_solutions=16):
    num_correct = len(correct_solutions)
    num_incorrect = len(incorrect_solutions)

    # We only want hard instances where incorrect solutions >= correct solutions
    if num_incorrect < num_correct:
        # Reduce the number of correct solutions to match the number of incorrect solutions
        random.shuffle(correct_solutions)
        correct_solutions = correct_solutions[:num_incorrect]
        num_correct = num_incorrect
    
    total_candidate_solutions = min(num_correct + num_incorrect, max_solutions)

    num_solutions = _generate_random_count(
        min_val=2, max_val=total_candidate_solutions, peak=(total_candidate_solutions + 2) // 2)
    
    cand_solutions, remaining_solutions = [], []
    # First add at least one correct and one incorrect solution
    # Add correct solution
    random.shuffle(correct_solutions)
    cand_solutions.append(('Correct', correct_solutions[0]))
    # Add incorrect solution
    random.shuffle(incorrect_solutions)
    cand_solutions.append(('Incorrect', incorrect_solutions[0]))

    # Add remaining solutions
    for solution in correct_solutions[1:]:
        remaining_solutions.append(('Correct', solution))
    for solution in incorrect_solutions[1:]:
        remaining_solutions.append(('Incorrect', solution))

    # Shuffle remaining solutions and add to candidate solutions till we reach num_solutions
    random.shuffle(remaining_solutions)
    cand_solutions.extend(remaining_solutions[:num_solutions - 2])

    # Shuffle candidate solutions
    random.shuffle(cand_solutions)

    # Create a consolidated instance
    instance = {
        "problem": correct_solutions[0]["problem"],
        "expected_answer": correct_solutions[0]["expected_answer"],
    }
    
    # Format the instance
    for i, (label, solution) in enumerate(cand_solutions):
        instance[f"solution_{i}"] = solution["generation"]
        instance[f"predicted_answer_{i}"] = solution["predicted_answer"]
        instance[f"label_{i}"] = label
    instance = _format_instance(instance=instance)
    return instance
