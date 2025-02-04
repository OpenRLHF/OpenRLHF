import torch


def reward_func(queries, prompts):
    print(queries)
    return torch.randn(len(queries))
