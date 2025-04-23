import torch

def experience_filter(experience_maker,experiences):
    return experiences[: len(experiences) // 2]