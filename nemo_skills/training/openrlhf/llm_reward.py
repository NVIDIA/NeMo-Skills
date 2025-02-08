import torch
from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics.utils import is_correct_judgement


def reward_func(queries: list[str], prompts: list[str], prompt_metadata: list[dict]):
    expected_answers = [data["expected_answer"] for data in prompt_metadata]
    predicted_answers = [extract_answer(query) for query in queries]
    problems = [data["problem"] for data in prompt_metadata]
    llm = get_model(server_type="trtllm")
    prompt = get_prompt('judge/math', 'qwen-instruct')
    prompts = [
        prompt.fill({'problem': problem, 'expected_answer': expected_answer, 'predicted_answer': predicted_answer})
        for problem, expected_answer, predicted_answer in zip(problems, expected_answers, predicted_answers)
    ]
    outputs = llm.generate(prompts=prompts)
    is_correct_array = [is_correct_judgement(output["generation"]) for output in outputs]

    return torch.tensor(is_correct_array, dtype=torch.float32)
