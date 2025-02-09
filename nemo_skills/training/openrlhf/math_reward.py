import json
import os

import torch

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import prefill_judgement


def reward_func(queries: list[str], prompts: list[str], prompt_metadata: list[dict]):
    data_points = []
    prefilled_judgements = []
    prefilled_indices = set()
    for metadata, query in zip(prompt_metadata, queries):
        data_points.append(
            {
                "problem": metadata["problem"],
                "expected_answer": metadata["expected_answer"],
                "predicted_answer": extract_answer(query),
            }
        )
        judgement = prefill_judgement(data_points[-1])
        if judgement is not None:
            prefilled_judgements.append(judgement)
            prefilled_indices.add(len(data_points) - 1)

    host = os.getenv("SLURM_MASTER_NODE_HET_GROUP_0", "localhost")
    server_args = json.loads(os.getenv("REWARD_SERVER_ARGS", "{}"))
    llm = get_model(server_type="trtllm", host=host, **server_args)
    prompt = get_prompt('judge/math', 'qwen-instruct')
    prompts = [prompt.fill(dp) for dp in data_points]
    outputs = llm.generate(prompts=prompts, stop_phrases=prompt.stop_phrases)
    judgements = []
    prefilled_idx = 0
    for idx, output in enumerate(outputs):
        if idx in prefilled_indices:
            judgements.append(prefilled_judgements[prefilled_idx])
            prefilled_idx += 1
        else:
            judgements.append(output["generation"])
    is_correct_array = [is_correct_judgement(judgement) for judgement in judgements]
    return torch.tensor(is_correct_array, dtype=torch.float32)
