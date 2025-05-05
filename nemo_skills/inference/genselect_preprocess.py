# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os 
import json
from collections import defaultdict
import logging
import random
import glob
import hydra
import math
from copy import deepcopy
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import nested_dataclass, setup_logging


LOG = logging.getLogger(__file__)


def read_file(file_path):
    LOG.info(f"Reading file: {file_path}")
    instances = [json.loads(line) for line in open(file_path, "r")]
    problem_to_instance = {instance["problem"]: instance for instance in instances}
    return problem_to_instance


def read_files(file_paths, single_answer_instances_path):
    problem_to_instances = defaultdict(list)
    for file_path in file_paths:
        problem_to_instance = read_file(file_path)
        for problem, instance in problem_to_instance.items():
            problem_to_instances[problem].append(instance)

    LOG.info(f"Number of problems: {len(problem_to_instances)}")

    with open(single_answer_instances_path, "w") as f:
        problem_to_clustered_instances = {}
        for problem, instance_list in problem_to_instances.items():
            answer_clusters = defaultdict(list)
            expected_answer = None
            for instance in instance_list:
                answer = instance["predicted_answer"]
                expected_answer = instance["expected_answer"]
                if answer is None:
                    continue
                answer_clusters[answer].append(instance)
            
            if len(answer_clusters) < 2:
                # Single answer or no answer
                if len(answer_clusters) == 1:
                    _, single_answer_instance_list = list(answer_clusters.items())[0]
                    instance = single_answer_instance_list[0]
                    single_answer_instance = deepcopy(instance)
                    single_answer_instance["is_correct"] = (is_correct_judgement(instance["judgement"]) if "judgement" in instance else instance["is_correct"])
                else:
                    single_answer_instance = {
                        "problem": problem, 
                        "predicted_answer": None, 
                        "expected_answer": expected_answer, 
                        "is_correct": False
                    }
                f.write(json.dumps(single_answer_instance) + "\n")
            else:
                problem_to_clustered_instances[problem] = [(answer, instances) for answer, instances in answer_clusters.items()]
        
    LOG.info(f"Number of problems with multiple answers: {len(problem_to_clustered_instances)}")
    return problem_to_clustered_instances


def extract_summary(solution, max_length=5000):
    """Extract the summary from the solution."""
    if solution.count("</think>") == 0:
        if len(solution) < max_length:
            # Probably the solution is a summary itself
            summary = solution
        else:
            # Take the last 10 steps
            summary = "\n\n".join(solution.split("\n\n")[-10:])[-max_length:]
    else:
        # There's a clear demarcation between the thinking step and the summary
        summary = solution.rsplit("</think>", 1)[1]
    
    summary = summary.replace("<think>", "")
    summary = summary.strip().rstrip("<|im_end|>")

    if len(summary) > max_length:
        summary = summary[-max_length:]
    return summary


def probabilistic_ceil(n: float) -> int:
    decimal_part = n - math.floor(n)
    if random.random() < decimal_part:
        return math.ceil(n)
    else:
        return math.floor(n)
    

def sample_instances(clustered_instances, max_soln_samples=8, sampling_strategy="linear", bayesian_constant=1.0):
    random.shuffle(clustered_instances)

    answer_counts = []
    for (_, same_answer_instances) in clustered_instances:
        answer_counts.append(len(same_answer_instances))

    total_samples = sum(answer_counts)    

    if sampling_strategy == "sqrt":
        unnormalized_sampling_probs = [(answer_count / total_samples) ** 0.5 for answer_count in answer_counts]
        sampling_probs = [sampling_prob / sum(unnormalized_sampling_probs) for sampling_prob in unnormalized_sampling_probs]

    elif sampling_strategy == "bayesian":
        pseudo_answer_counts = [(answer_count + bayesian_constant) for answer_count in answer_counts]
        sampling_probs = [
            pseudo_answer_count / sum(pseudo_answer_counts) for pseudo_answer_count in pseudo_answer_counts]
    else:
        sampling_probs = [answer_count / total_samples for answer_count in answer_counts]
      
    # Sample instances from each cluster using the sampling probabilities   
    sampled_instances = []
    num_samples = min(max_soln_samples, total_samples)
    for i, (_, same_answer_instances) in enumerate(clustered_instances):
        cur_num_samples = probabilistic_ceil(sampling_probs[i] * num_samples)
        cur_num_samples = min(max(1, cur_num_samples), len(same_answer_instances))
        # if cur_num_samples > 0:
        sampled_instances.extend(random.sample(same_answer_instances, cur_num_samples))

    return sampled_instances[:max_soln_samples]


def create_comparison_instance(clustered_instances, problem, max_soln_samples=8, sampling_strategy="linear"):
    # Create a consolidated instance
    comparison_instance = {
        "problem": problem,
    }
    
    sampled_instances = sample_instances(clustered_instances, max_soln_samples=max_soln_samples, sampling_strategy=sampling_strategy)
    sampled_solutions = [extract_summary(instance["generation"]) for instance in sampled_instances]
    consolidated_solutions = ""
    for idx, solution in enumerate(sampled_solutions):
        consolidated_solutions += f"Solution {idx}:\n{solution}\n\n"

    comparison_instance["solutions"] = consolidated_solutions
    comparison_instance["max_idx"] = len(sampled_solutions) - 1
    comparison_instance["num_solutions"] = len(sampled_instances)

    for i, instance in enumerate(sampled_instances):
        comparison_instance[f"predicted_answer_{i}"] = instance["predicted_answer"]
        if "judgement" in instance:
            comparison_instance[f"is_correct_{i}"] =  is_correct_judgement(instance["judgement"])
        elif "is_correct" in instance:
            comparison_instance[f"is_correct_{i}"] = instance["is_correct"]
        else:
            comparison_instance[f"is_correct_{i}"] = (instance["predicted_answer"] == instance["expected_answer"])

    comparison_instance["expected_answer"] = clustered_instances[0][1][0]["expected_answer"]

    return comparison_instance


def preprocess(input_dir, output_dir, max_soln_samples=8, sampling_strategy="linear", num_random_seeds=8, num_input_samples=8):    
    if output_dir is None:
        raise ValueError("Output directory is required")
    
    output_dir = os.path.join(output_dir, "comparison_instances")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(input_dir, "output-rs*.jsonl")))
    if num_input_samples is not None:
        input_files = input_files[:num_input_samples]
        print(f"Using {num_input_samples} / {len(input_files)} input files")
    problem_to_clustered_instances = read_files(input_files, os.path.join(output_dir, "single_answer_instances.jsonl"))
    
    for random_seed in range(num_random_seeds):
        # random.seed(random_seed)
        with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
            for problem, clustered_instances in problem_to_clustered_instances.items():
                comparison_instance = create_comparison_instance(clustered_instances, problem, max_soln_samples=max_soln_samples, sampling_strategy=sampling_strategy)
                f.write(json.dumps(comparison_instance) + "\n")



@nested_dataclass(kw_only=True)
class GenSelectPreprocessConfig:
    input_dir: str
    output_dir: str
    max_soln_samples: int = 16
    sampling_strategy: str = "linear"
    num_random_seeds: int | None = None
    num_input_samples: int | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_preprocess_config", node=GenSelectPreprocessConfig)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_preprocess_config')
def genselect_preprocessor(cfg: GenSelectPreprocessConfig):
    cfg = GenSelectPreprocessConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    preprocess(input_dir=cfg.input_dir, output_dir=cfg.output_dir, max_soln_samples=cfg.max_soln_samples, 
               sampling_strategy=cfg.sampling_strategy, num_random_seeds=cfg.num_random_seeds, num_input_samples=cfg.num_input_samples)


if __name__ == "__main__":
    setup_logging()
    genselect_preprocessor()