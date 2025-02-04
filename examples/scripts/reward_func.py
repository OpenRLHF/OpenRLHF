import torch


def reward_func(queries, prompts):
    # queries is prompts + responses
    print(queries)
    return torch.randn(len(queries))
