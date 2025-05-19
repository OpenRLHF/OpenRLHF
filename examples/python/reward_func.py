import torch


def reward_func(queries, prompts, labels, **kwargs):
    # queries is prompts + responses
    # labels is answers
    print(queries)
    return torch.randint(0, 2, (len(queries),)).float()
