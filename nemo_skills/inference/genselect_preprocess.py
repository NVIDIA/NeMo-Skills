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
import importlib

import hydra

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class GenSelectPreprocessConfig:
    input_dir: str
    output_dir: str | None = None
    benchmark: str
    input_key: str
    output_key: str
    cluster_key: str | None = None
    max_soln_samples: int = 16
    is_competition: bool = False
    sampling_strategy: str = "linear"
    num_random_seeds: int | None = None
    num_input_samples: int | None = None


def probabilistic_ceil(n: float) -> int:
    decimal_part = n - math.floor(n)
    if random.random() < decimal_part:
        return math.ceil(n)
    else:
        return math.floor(n)


class GenSelectPreprocessor:
    def __init__(self, cfg: GenSelectPreprocessConfig):
        self.cfg = cfg
        self.input_dir = cfg.input_dir
        self.output_dir = cfg.output_dir
        self.benchmark = cfg.benchmark
        self.input_key = cfg.input_key
        self.output_key = cfg.output_key
        self.cluster_key = cfg.cluster_key  # Key to cluster instances by
        self.max_soln_samples = cfg.max_soln_samples
        self.sampling_strategy = cfg.sampling_strategy
        self.num_random_seeds = cfg.num_random_seeds
        self.num_input_samples = cfg.num_input_samples

        # Initialize the class
        self._post_init()
        
    def _post_init(self):
        dataset_module = None
        try:
            dataset_module = importlib.import_module(f"nemo_skills.dataset.{self.benchmark}")
        except ImportError:
            LOG.warning(f"Dataset module {self.benchmark} not found. Ignoring the use of associated metric.")
        
        if dataset_module is not None:
            metric_type = dataset_module.METRICS_TYPE
            from nemo_skills.evaluation.metrics.map_metrics import METRICS_MAP
            metric_module = METRICS_MAP[metric_type]
            self.get_score_dict = metric_module._get_score_dict
        else:
            self.get_score_dict = None

    def get_instance_correctness(self, instance):
        if self.get_score_dict is None:
            if "accuracy" in instance:
                return bool(instance["accuracy"])
            elif "judge_correct" in instance:
                return bool(instance["judge_correct"])
        else:
            score_dict = self.get_score_dict(instance)
            if len(score_dict) == 1:
                # Just one score, so we can use it to determine correctness
                return bool(list(score_dict.values())[0])
            else:
                # Multiple scores, not clear how to determine correctness
                # If all scores are the same, then we can use that value
                # Otherwise we can return None
                if len(set(score_dict.values())) == 1:
                    return bool(list(score_dict.values())[0])
                else:
                    return None
    
    def read_file(self, file_path):
        LOG.info(f"Reading file: {file_path}")
        instances = [json.loads(line) for line in open(file_path, "r")]
        problem_to_instance = {instance[self.input_key]: instance for instance in instances}
        return problem_to_instance
                
    def read_files(self, file_paths, single_correctness_instances_path):
        problem_to_instances = defaultdict(list)
        for file_path in file_paths:
            problem_to_instance = self.read_file(file_path)
            for problem, instance in problem_to_instance.items():
                problem_to_instances[problem].append(instance)

        LOG.info(f"Number of problems: {len(problem_to_instances)}")

        # Identify all instances which have the same correctness value
        rem_problems = []
        with open(single_correctness_instances_path, "w") as f:
            for problem, instance_list in problem_to_instances.items():
                correctness_vals = set([self.get_instance_correctness(instance) for instance in instance_list])
                if len(correctness_vals) == 1 and None not in correctness_vals:
                    # Single correctness
                    f.write(json.dumps(instance_list[0]) + "\n")
                    continue
                else:
                    # Need to cluster these instances
                    rem_problems.append(problem)
        
        # Now cluster the instances by the cluster key
        problem_to_clustered_instances = {}
        if self.cluster_key is not None:
            for problem in rem_problems:
                instance_list = problem_to_instances[problem]
                cluster_dict = defaultdict(list)
                for instance in instance_list:
                    cluster_key_val = instance[self.cluster_key]
                    cluster_dict[cluster_key_val].append(instance)
                
                problem_to_clustered_instances[problem] = [instance_list for _, instance_list in cluster_dict.items()]
        else:
            problem_to_clustered_instances[problem] = [[instance] for instance in problem_to_instances]

        LOG.info(f"Number of problems passed to GenSelect: {len(problem_to_clustered_instances)}")
        return problem_to_clustered_instances
    

    def sample_instances(self, clustered_instances):
        random.shuffle(clustered_instances)

        cluster_counts = []
        for _, same_cluster_instances in clustered_instances:
            cluster_counts.append(len(same_cluster_instances))
        total_samples = sum(cluster_counts)

        if self.sampling_strategy == "sqrt":
            unnormalized_sampling_probs = [(cluster_count / total_samples) ** 0.5 for cluster_count in cluster_counts]
            sampling_probs = [
                sampling_prob / sum(unnormalized_sampling_probs) for sampling_prob in unnormalized_sampling_probs
            ]
        else:
            sampling_probs = [cluster_count / total_samples for cluster_count in cluster_counts]

        # Sample instances from each cluster using the sampling probabilities
        sampled_instances = []
        num_samples = min(self.max_soln_samples, total_samples)
        for i, (_, same_cluster_instances) in enumerate(clustered_instances):
            cur_num_samples = probabilistic_ceil(sampling_probs[i] * num_samples)
            cur_num_samples = min(max(1, cur_num_samples), len(same_cluster_instances))
            sampled_instances.extend(random.sample(same_cluster_instances, cur_num_samples))

        return sampled_instances[:self.max_soln_samples]


    def create_comparison_instance(self, clustered_instances):
        # Create a consolidated instance
        sampled_instances = self.sample_instances(clustered_instances)
        sampled_solutions = [instance[self.output_key] for instance in sampled_instances]
        consolidated_solutions = ""
        for idx, solution in enumerate(sampled_solutions):
            consolidated_solutions += f"Solution {idx}:\n{solution}\n\n"

        comparison_instance = deepcopy(sampled_instances[0])
        comparison_instance["solutions"] = consolidated_solutions
        comparison_instance["max_idx"] = len(sampled_solutions) - 1
        comparison_instance["num_solutions"] = len(sampled_instances)

        for i, instance in enumerate(sampled_instances):
            comparison_instance[f"{self.output_key}_{i}"] = instance[self.output_key]

        return comparison_instance

    def preprocess(self):
        if self.output_dir is None:
            raise ValueError("Output directory is required")

        output_dir = os.path.join(self.output_dir, "comparison_instances")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        input_files = sorted(glob.glob(os.path.join(self.input_dir, "output-rs*.jsonl")))
        if self.num_input_samples is not None:
            input_files = input_files[:self.num_input_samples]
            LOG.info(f"Using {self.num_input_samples} / {len(input_files)} input files")
        
        problem_to_clustered_instances = self.read_files(
            input_files, os.path.join(output_dir, "single_correctness_instances.jsonl")
        )

        for random_seed in range(self.num_random_seeds):
            random.seed(random_seed)
            with open(os.path.join(output_dir, f"output-rs{random_seed}.jsonl"), "w") as f:
                for problem, clustered_instances in problem_to_clustered_instances.items():
                    comparison_instance = self.create_comparison_instance(clustered_instances)
                    f.write(json.dumps(comparison_instance) + "\n")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_genselect_preprocess_config", node=GenSelectPreprocessConfig)


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_genselect_preprocess_config')
def genselect_preprocessor(cfg: GenSelectPreprocessConfig):
    cfg = GenSelectPreprocessConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    genselect_preprocessor = GenSelectPreprocessor(cfg)
    genselect_preprocessor.preprocess()


if __name__ == "__main__":
    setup_logging()
    genselect_preprocessor()
