import torch


def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    # labels is answers
    print(queries)
    return torch.randn(len(queries))
