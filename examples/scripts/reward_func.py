import torch


def reward_func(prompts):
    return torch.randn(len(prompts))
