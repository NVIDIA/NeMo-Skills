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
import re
import hydra

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def read_file(file_path):
    LOG.info(f"Reading file: {file_path}")
    instances = [json.loads(line) for line in open(file_path, "r")]
    for instance in instances:
        if "problem" not in instance:
            if "question" in instance:
                instance["problem"] = instance["question"]

        # Overwrite is_correct if graded_list or judgement is present
        if "graded_list" in instance:
            instance["is_correct"] = instance["graded_list"][0]
        if "judgement" in instance:
            instance["is_correct"] = is_correct_judgement(instance["judgement"])


        if "completion" in instance:
            instance["generation"] = instance["completion"]
            instance["predicted_answer"] = instance["completion"]

    problem_to_instance = {instance["problem"]: instance for instance in instances}
    return problem_to_instance


def check_if_all_incorrect(answer_clusters):
    # Check if all answers are incorrect
    all_incorrect = True
    for answer, instances in answer_clusters.items():
        if answer is None:
            continue
        else:
            instance = instances[0]
            if "judgement" in instance:
                if is_correct_judgement(instance["judgement"]):
                    all_incorrect = False
                    break
            else:
                if instance["is_correct"]:
                    all_incorrect = False
                    break

    # If all answers are incorrect, just choose the most common answer
    if all_incorrect:
        # Choose the most common answer
        most_common_answer = max(answer_clusters, key=lambda x: len(answer_clusters[x]))
        return answer_clusters[most_common_answer][0]
    else:
        return None


def check_for_single_viable_answer(answer_clusters):
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
        return single_answer_instance

    elif (len(answer_clusters) == 2 and (None in list(answer_clusters.keys()))):
        # Only one real answer because the other one is None
        for answer, instances in answer_clusters.items():
            if answer is None:
                continue
            else:
                instance = instances[0]
                single_answer_instance = deepcopy(instance)
                single_answer_instance["is_correct"] = (
                    is_correct_judgement(instance["judgement"])
                    if "judgement" in instance
                    else instance["is_correct"]
                )
                return single_answer_instance
    else:
        return None


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

            single_answer_instance = check_for_single_viable_answer(answer_clusters)
            if single_answer_instance is not None:
                LOG.warning(f"Single answer instance found for problem {problem}")
                f.write(json.dumps(single_answer_instance) + "\n")
            else:
                # Check if all answers are incorrect
                instance = check_if_all_incorrect(answer_clusters)
                if instance is not None:
                    LOG.warning(f"All incorrect answer instance found for problem {problem}")
                    f.write(json.dumps(instance) + "\n")
                else:
                    # Write down the instances for problems with multiple answers
                    problem_to_clustered_instances[problem] = [
                        (answer, instances) for answer, instances in answer_clusters.items()
                    ]

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
            new_instance = {"problem": instance["problem"]}
            for key in ["problem", "id", "subset_for_metrics"]:
                if key in instance:
                    new_instance[key] = instance[key]

            if instance["judgment_idx"] is not None:
                judgment_idx = instance["judgment_idx"]
            else:
                judgment_idx = random.randint(0, instance["max_idx"])

            new_instance["generation"] = instance[f"solution_{judgment_idx}"]
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



def minibatchify_instances(clustered_instances, max_soln_samples=8, use_diversity=False):
    if not use_diversity:
        # Original behavior
        instances = []
        for _, same_answer_instances in clustered_instances:
            instances.extend(same_answer_instances)
        random.shuffle(instances)
        minibatch_instances = [instances[i:i+max_soln_samples] for i in range(0, len(instances), max_soln_samples)]
        return minibatch_instances
    
    else:
        # Diversity-based distribution
        # Sort clusters by size (smallest first)
        sorted_clusters = sorted(clustered_instances, key=lambda x: len(x[1]))
        
        # Create ordered list prioritizing diversity
        distributed_instances = []
        for _, same_answer_instances in sorted_clusters:
            shuffled_instances = same_answer_instances.copy()
            random.shuffle(shuffled_instances)
            distributed_instances.extend(shuffled_instances)
        
        # Create minibatches using round-robin distribution
        num_batches = (len(distributed_instances) + max_soln_samples - 1) // max_soln_samples
        minibatch_instances = [[] for _ in range(num_batches)]
        
        for i, instance in enumerate(distributed_instances):
            minibatch_instances[i % num_batches].append(instance)

        # Shuffle the individual minibatches in place
        for minibatch in minibatch_instances:
            random.shuffle(minibatch)
        
        return minibatch_instances


def create_comparison_instance(clustered_instances, max_soln_samples=8, use_diversity=False):
    """Create a comparison instance for a problem."""
    # Create a consolidated instance
    all_minibatch_instances = minibatchify_instances(clustered_instances, max_soln_samples, use_diversity)

    comparison_instances = []
    for minibatch_instances in all_minibatch_instances:
        sampled_solutions = [instance["generation"] for instance in minibatch_instances]

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

        for key in ["generation", "judgement", "tokens", "logprobs", "generation_time", "stopped_on_repetition", "is_new_summary_longer", "is_correct", "solutions"]:
            if key in comparison_instance:
                del comparison_instance[key]
        
        comparison_instance["expected_answer"] = minibatch_instances[0]["expected_answer"]
        comparison_instances.append(comparison_instance)

    return comparison_instances


def process_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds, use_diversity):
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
                    use_diversity=use_diversity,
                )
                for comparison_instance in comparison_instances:
                    f.write(json.dumps(comparison_instance) + "\n")


def process_non_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds, use_diversity):
    problem_to_clustered_instances = read_files(input_files, os.path.join(output_dir, "single_answer_instances.jsonl"))

    for random_seed in range(num_random_seeds):
        # random.seed(random_seed)
        with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
            for _, clustered_instances in problem_to_clustered_instances.items():
                comparison_instances = create_comparison_instance(
                    clustered_instances,
                    max_soln_samples=max_soln_samples,
                    use_diversity=use_diversity,
                )
                for comparison_instance in comparison_instances:
                    f.write(json.dumps(comparison_instance) + "\n")


def preprocess(
    input_dir, output_dir, max_soln_samples=8, num_random_seeds=8, num_input_samples=8, use_diversity=False
):
    if output_dir is None:
        raise ValueError("Output directory is required")

    output_dir = os.path.join(output_dir, f"comparison_instances")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Extract numeric index from filename for proper sorting
    def extract_index(filename):
        match = re.search(r'output-rs(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0

    input_files = sorted(glob.glob(os.path.join(input_dir, "output-rs*.jsonl")), key=extract_index)

    if num_input_samples is not None:
        input_files = input_files[:num_input_samples]
        print(f"Using {num_input_samples} / {len(input_files)} input files")

    in_competition = test_if_in_competition(input_files[0])
    
    if in_competition:
        process_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds, use_diversity)
    else:
        process_non_competition_files(input_files, output_dir, max_soln_samples, num_random_seeds, use_diversity)


@nested_dataclass(kw_only=True)
class GenSelectPreprocessConfig:
    input_dir: str
    output_dir: str
    max_soln_samples: int = 16
    num_random_seeds: int | None = None
    num_input_samples: int | None = None
    use_diversity: bool = False


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
        use_diversity=cfg.use_diversity,
    )


if __name__ == "__main__":
    setup_logging()
    genselect_preprocessor()
    
