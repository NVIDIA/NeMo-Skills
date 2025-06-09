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

import glob
import json
import logging
import math
import os
import random
from collections import defaultdict
from copy import deepcopy

import hydra

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def read_file(file_path):
    LOG.info(f"Reading file: {file_path}")
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            # if "hmmt_F_2024" in instance["id"]:
            # if "aime25" in instance["id"]:
            if True:
                instances.append(instance)

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
            for instance in instance_list:
                answer = instance["predicted_answer"]
                answer_clusters[answer].append(instance)

            if len(answer_clusters) == 1:
                # Single answer or no answer
                _, single_answer_instance_list = list(answer_clusters.items())[0]
                instance = single_answer_instance_list[0]
                single_answer_instance = deepcopy(instance)
                if single_answer_instance["predicted_answer"] is None:
                    # The only predicted answer across seeds is None
                    single_answer_instance["is_correct"] = False
                else:
                    single_answer_instance["is_correct"] = (
                        is_correct_judgement(instance["judgement"])
                        if "judgement" in instance
                        else instance["is_correct"]
                    )

                f.write(json.dumps(single_answer_instance) + "\n")
            else:
                problem_to_clustered_instances[problem] = [
                    (answer, instances) for answer, instances in answer_clusters.items()
                ]

    LOG.info(f"Number of problems with multiple answers: {len(problem_to_clustered_instances)}")
    return problem_to_clustered_instances



def test_if_in_competition(file_path):
    instances = [json.loads(line) for line in open(file_path, "r")]
    instances = [instance for instance in instances if "judgment_idx" in instance]
    if len(instances) > 0:
        return True
    else:
        return False


def read_file_competition(file_path, single_answer_instances_path=None):
    instances = []
    for line in open(file_path, "r"):
        instance = json.loads(line)
        
        if "judgment_idx" in instance:
            new_instance = {"problem": instance["problem"], "expected_answer": instance["expected_answer"]}
            for key in ["problem", "expected_answer", "id", "subset_for_metrics", "reference_solution"]:
                if key in instance:
                    new_instance[key] = instance[key]

            if instance["judgment_idx"] is not None:
                judgment_idx = instance["judgment_idx"]
            else:
                judgment_idx = random.randint(0, instance["max_idx"])

            new_instance["generation"] = instance[f"solution_{judgment_idx}"]
            new_instance["summary"] = instance[f"solution_{judgment_idx}"]
            new_instance["is_correct"] = instance[f"is_correct_{judgment_idx}"]
            new_instance["predicted_answer"] = instance[f"predicted_answer_{judgment_idx}"]
            new_instance["judgement"] = instance[f"judgement_{judgment_idx}"]

            instances.append(new_instance)
        else:
            instances.append(instance)

    
    problem_to_instances = defaultdict(list)
    for instance in instances:
        problem_to_instances[instance["problem"]].append(instance)

    non_single_problem_to_instances = {problem: instances for problem, instances in problem_to_instances.items() if len(instances) > 1}
    single_problem_to_instances = {problem: instances[0] for problem, instances in problem_to_instances.items() if len(instances) == 1}
    
    if single_answer_instances_path is not None:
        with open(single_answer_instances_path, "w") as f:
            for _, instance in single_problem_to_instances.items():
                f.write(json.dumps(instance) + "\n")
    
    LOG.warning(f"Number of problems with multiple answers: {len(non_single_problem_to_instances)}")

    problem_to_clustered_instances = {}
    for problem, instance_list in non_single_problem_to_instances.items():
        answer_clusters = defaultdict(list)
        for instance in instance_list:
            answer = instance["predicted_answer"]
            answer_clusters[answer].append(instance)

        problem_to_clustered_instances[problem] = [
            (answer, instances) for answer, instances in answer_clusters.items()
        ]
    
    return problem_to_clustered_instances


def extract_summary(instance, max_length=5000):
    """Extract the summary from the solution."""
    if "summary" in instance:
        return instance["summary"]

    solution = instance["generation"]
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

    if len(summary) > max_length:
        summary = summary[-max_length:]
    return summary.strip()




def minibatchify_instances(clustered_instances, max_soln_samples=8):
    instances = []
    for _, same_answer_instances in clustered_instances:
        instances.extend(same_answer_instances)
    random.shuffle(instances)

    minibatch_instances = [instances[i:i+max_soln_samples] for i in range(0, len(instances), max_soln_samples)]
    return minibatch_instances

def create_comparison_instance(clustered_instances, max_soln_samples=8):
    """Create a comparison instance for a problem."""
    # Create a consolidated instance
    all_minibatch_instances = minibatchify_instances(clustered_instances, max_soln_samples)

    comparison_instances = []
    for minibatch_instances in all_minibatch_instances:
        sampled_solutions = [extract_summary(instance) for instance in minibatch_instances]

        comparison_instance = deepcopy(minibatch_instances[0])
        consolidated_solutions = ""
        for idx, solution in enumerate(sampled_solutions):
            consolidated_solutions += f"Solution {idx}:\n{solution}\n\n"
            comparison_instance[f"solution_{idx}"] = solution

        comparison_instance["solutions"] = consolidated_solutions
        comparison_instance["max_idx"] = len(sampled_solutions) - 1
        comparison_instance["num_solutions"] = len(sampled_solutions)

        for i, instance in enumerate(minibatch_instances):
            comparison_instance[f"predicted_answer_{i}"] = instance["predicted_answer"]
            if "judgement" in instance:
                comparison_instance[f"judgement_{i}"] = instance["judgement"]
            if "is_correct" in instance:
                comparison_instance[f"is_correct_{i}"] = instance["is_correct"]

        for key in ["generation", "judgement", "tokens", "logprobs", "generation_time", "stopped_on_repetition", "is_new_summary_longer", "is_correct"]:
            if key in comparison_instance:
                del comparison_instance[key]
        
        comparison_instance["expected_answer"] = minibatch_instances[0]["expected_answer"]
        comparison_instances.append(comparison_instance)

    return comparison_instances


def process_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds):
    for random_seed, input_file in enumerate(input_files):
        if random_seed == 0:
            problem_to_clustered_instances = read_file_competition(input_file, os.path.join(output_dir, "single_answer_instances.jsonl"))
        else:
            problem_to_clustered_instances = read_file_competition(input_file)

        with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
            for _, clustered_instances in problem_to_clustered_instances.items():
                comparison_instances = create_comparison_instance(
                    clustered_instances,
                    max_soln_samples=max_soln_samples,
                )
                for comparison_instance in comparison_instances:
                    f.write(json.dumps(comparison_instance) + "\n")


def process_non_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds):
    problem_to_clustered_instances = read_files(input_files, os.path.join(output_dir, "single_answer_instances.jsonl"))

    for random_seed in range(num_random_seeds):
        # random.seed(random_seed)
        with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
            for _, clustered_instances in problem_to_clustered_instances.items():
                comparison_instances = create_comparison_instance(
                    clustered_instances,
                    max_soln_samples=max_soln_samples,
                )
                for comparison_instance in comparison_instances:
                    f.write(json.dumps(comparison_instance) + "\n")


def preprocess(
    input_dir, output_dir, max_soln_samples=8, num_random_seeds=8, num_input_samples=8
):
    if output_dir is None:
        raise ValueError("Output directory is required")

    output_dir = os.path.join(output_dir, f"comparison_instances")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(input_dir, "output-rs*.jsonl")))
    if num_input_samples is not None:
        input_files = input_files[:num_input_samples]
        print(f"Using {num_input_samples} / {len(input_files)} input files")

    in_competition = test_if_in_competition(input_files[0])
    
    if in_competition:
        process_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds)
    else:
        process_non_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds)




@nested_dataclass(kw_only=True)
class GenSelectPreprocessConfig:
    input_dir: str
    output_dir: str
    max_soln_samples: int = 16
    num_random_seeds: int | None = None
    num_input_samples: int | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_preprocess_config", node=GenSelectPreprocessConfig)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_preprocess_config')
def genselect_preprocessor(cfg: GenSelectPreprocessConfig):
    cfg = GenSelectPreprocessConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    preprocess(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        max_soln_samples=cfg.max_soln_samples,
        num_random_seeds=cfg.num_random_seeds,
        num_input_samples=cfg.num_input_samples,
    )


if __name__ == "__main__":
    setup_logging()
    genselect_preprocessor()
    

