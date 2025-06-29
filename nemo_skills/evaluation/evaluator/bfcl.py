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

# adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py

import inspect
import json
import logging
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from nemo_skills.inference.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLEvaluatorConfig:
    model: str = "Qwen/Qwen3-4B"
    timeout: int = 300


def eval_bfcl(cfg):
    """BFCL (Berkeley Function Calling Leaderboard) evaluation wrapper.
    
    This function wraps the external BFCL evaluation tool, converting between
    NeMo-Skills format and BFCL format, then merging results back.
    """
    eval_config = BFCLEvaluatorConfig(**cfg.eval_config)
    model_name = eval_config.model.replace('/', '_')
    # model_name = eval_config.model.split("/")[-1]
    for jsonl_file in unroll_files(cfg.input_files):
        parent_dir = Path(jsonl_file).absolute().parent
        # Test categories are labeled as bfcl.simple, bfcl.parallel, bfcl.multiple
        test_category = Path(parent_dir).name.split(".")[1]
        
        # Convert NeMo-Skills format to BFCL format
        output_dir = Path("/opt/gorilla/berkeley-function-call-leaderboard") / f"result/{model_name}"
        bfcl_input_file = _convert_to_bfcl_format(jsonl_file, output_dir=output_dir, test_category=test_category)

        # Copy the bfcl_input_file to the output_dir
        common_bfcl_output_dir = Path(parent_dir).parent / Path(model_name)
        if not common_bfcl_output_dir.exists():
            common_bfcl_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(bfcl_input_file, common_bfcl_output_dir / Path(bfcl_input_file).name)

        try:
            # Run BFCL evaluation using the CLI
            cmd = (
                f'bfcl evaluate --model {eval_config.model} '
                f'--test-category {test_category}'
            )
            
            LOG.info(f"Running BFCL evaluation: {cmd}")
            subprocess.run(cmd, shell=True, check=True, timeout=eval_config.timeout)
            
            # Merge BFCL results back into original file
            # _merge_bfcl_results(jsonl_file, parent_dir, eval_config.test_categories)
            
        except subprocess.TimeoutExpired:
            LOG.error(f"BFCL evaluation timed out after {eval_config.timeout} seconds")
            raise


def _convert_to_bfcl_format(jsonl_file, output_dir, test_category):
    """Convert NeMo-Skills JSONL format to BFCL expected format."""
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    bfcl_file = Path(output_dir, f"BFCL_v3_{test_category}_result.json")

    with open(jsonl_file, 'rt', encoding='utf-8') as fin, \
         open(bfcl_file, 'wt', encoding='utf-8') as fout:
        
        for line in fin:
            sample = json.loads(line)
            
            # Convert to BFCL format - adjust based on actual BFCL input requirements
            bfcl_sample = {
                'id': sample.get('id', sample.get('problem_id', '')),
                'result': sample.get('generation', ''),
            }
                            
            fout.write(json.dumps(bfcl_sample) + '\n')
    
    return bfcl_file


def _merge_bfcl_results(jsonl_file, parent_dir, test_categories):
    """Merge BFCL evaluation results back into the original NeMo-Skills file."""
    # Read original data
    with open(jsonl_file, 'rt', encoding='utf-8') as fin:
        samples = [json.loads(line) for line in fin]
    
    # Find and read BFCL result files
    categories = test_categories.split(',')
    for category in categories:
        result_file = parent_dir / f'BFCL_v3_{category.strip()}_score.json'
        if result_file.exists():
            with open(result_file, 'rt', encoding='utf-8') as fin:
                bfcl_results = json.load(fin)
            
            # Merge results back into samples
            for i, sample in enumerate(samples):
                sample_id = sample.get('id', sample.get('problem_id', str(i)))
                if sample_id in bfcl_results:
                    sample[f'bfcl_{category}_result'] = bfcl_results[sample_id]
                    # Add standardized correctness keys for metrics system
                    if 'overall_accuracy' in bfcl_results[sample_id]:
                        sample['is_correct'] = bfcl_results[sample_id]['overall_accuracy'] > 0
            
            # Clean up BFCL result file
            result_file.unlink()
    
    # Write merged results back
    with open(jsonl_file, 'wt', encoding='utf-8') as fout:
        for sample in samples:
            fout.write(json.dumps(sample) + '\n')