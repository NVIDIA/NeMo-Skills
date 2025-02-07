import torch


def reward_func(queries: list[str], prompts: list[str]):
    print("GETTING CUSTOM REWARD")
    return torch.tensor([1.0 for _ in range(len(queries))])
