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


"""Script to prepare labeling data for GenSelect"""


import os 
import json
from collections import defaultdict
from utils import create_comparison_instance, segregate_instances
import logging
import random
import argparse
from transformers import AutoTokenizer
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from nemo_skills.utils import unroll_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import hashlib

# Skip if the solutions are too long
SKIP_LENGTH = 100_000
# We can skip tokenization if the solutions are shorter than this in character length
SAFE_SOLNS_LENGTH = 60_000
# Maximum number of tokens in the solutions
# This is based on the Qwen/QwQ-32B model. We can change this if we use a different model.
# The rough estimate is 20K tokens for solutions + less than 4K tokens for problem + prompt, will allow for 16K tokens for the solution. 
MAX_TOKEN_LENGTH = 20_000
MODEL_NAME = "Qwen/QwQ-32B"



_TOKENIZER = None
def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _TOKENIZER


def read_data(file_paths):
    problem_to_instances = defaultdict(list)
    for file_path in unroll_files(file_paths.split(",")):
        with open(file_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                problem = instance["problem"]
                problem_to_instances[problem].append(instance)
            
    logger.warning(f"Number of problems: {len(problem_to_instances)}")
    average_num_instances = sum([len(instances) for instances in problem_to_instances.values()]) / len(problem_to_instances)
    logger.warning(f"Average number of instances: {average_num_instances}")
    return problem_to_instances


def hash_signature(problem, solutions):
    text = (problem + solutions).encode('utf-8')
    return hashlib.md5(text).hexdigest()


def process_problem_batch(problem_batch, max_instances_per_problem, max_solutions, seed):
    """
    Process a *batch* of problems in a single worker call.
    """
    random.seed(seed)
    tokenizer = get_tokenizer()

    processed_results = []
    for (_, problem_instances) in problem_batch:
        # Segregate solutions into correct and incorrect
        correct_solutions, incorrect_solutions = segregate_instances(problem_instances)
        # Skip if no correct or no incorrect
        if not correct_solutions or not incorrect_solutions:
            continue
        # Skip if too few solutions
        if len(correct_solutions) + len(incorrect_solutions) < 4:
            continue

        unique_comparison_instances = set()
        problem_results = []

        for _ in range(2 * max_instances_per_problem):
            comparison_instance = create_comparison_instance(
                correct_solutions, incorrect_solutions, max_solutions=max_solutions
            )
            if comparison_instance is None:
                break

            # Check length first
            if len(comparison_instance["solutions"]) > SKIP_LENGTH:
                # skip
                continue

            signature = hash_signature(
                comparison_instance["problem"],
                comparison_instance["solutions"]
            )
            if signature in unique_comparison_instances:
                continue

            # Possibly do the token check if length ~ borderline
            if len(comparison_instance["solutions"]) < SAFE_SOLNS_LENGTH:
                unique_comparison_instances.add(signature)
                problem_results.append(comparison_instance)
            else:
                # actually tokenize if we might be under the threshold
                if len(tokenizer.encode(comparison_instance["solutions"])) <= MAX_TOKEN_LENGTH:
                    unique_comparison_instances.add(signature)
                    problem_results.append(comparison_instance)

            if len(problem_results) >= max_instances_per_problem:
                break

        processed_results.extend(problem_results)
    return processed_results


def prepare_data(
        input_files, max_instances_per_problem=4, max_solutions=16, seed=10, 
        num_workers=None, chunk_size=1000):
    if num_workers is None:
        num_workers = mp.cpu_count()
    random.seed(seed)

    # read file
    problem_to_instances = read_data(input_files)
    problems = list(problem_to_instances.items())

    # chunk the problems
    problem_chunks = [
        problems[i:i+chunk_size] for i in range(0, len(problems), chunk_size)
    ]
    
    all_unique_instances = []
    with ProcessPoolExecutor(max_workers=min(num_workers, len(problem_chunks))) as executor:
        # map over chunks
        results_iter = executor.map(
            partial(process_problem_batch, 
                    max_instances_per_problem=max_instances_per_problem,
                    max_solutions=max_solutions, 
                    seed=seed),
            problem_chunks
        )
        for chunk_idx, chunk_result in enumerate(results_iter, start=1):
            all_unique_instances.extend(chunk_result)
            if chunk_idx % 10 == 0:
                logger.warning(f"Processed chunk {chunk_idx}/{len(problem_chunks)}")

    logger.warning(f"Total unique instances: {len(all_unique_instances)}")
    return all_unique_instances


def save_data(unique_instances, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"output.jsonl"), "w") as f:
        for instance in unique_instances:
            f.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--max_instances_per_problem", type=int, default=8, 
                        help="Maximum number of GenSelect instances per problem")
    parser.add_argument("--max_solutions", type=int, default=16, 
                        help="Maximum number of solutions that form the GenSelect input")
    
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    unique_instances = prepare_data(
        args.input_files, args.max_instances_per_problem, args.max_solutions, args.seed)
    save_data(unique_instances, args.output_dir)
