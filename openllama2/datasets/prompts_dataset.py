from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data):
    # Dahoas/full-hh-rlhf
    if exist_and_not_none(data, 'prompt'):
        prompt = data['prompt']
        # tasksource/oasst1_pairwise_rlhf_reward
        if prompt.startswith('prompter:'):
            prompt = prompt.replace('prompter:', '\nHuman:').replace('assistant:', '\nAssistant:') + '\nAssistant: '
    # BelleGroup/train_0.5M_CN
    elif exist_and_not_none(data, 'instruction'):
        prompt = 'Human: ' +  data['instruction'] + "\nAssistant: "
    # stanfordnlp/SHP
    elif exist_and_not_none(data, 'history'):
        prompt = "Human: " +  data['history'] + "\nAssistant: "
    # lvwerra/stack-exchange-paired
    elif exist_and_not_none(data, 'question'):
        prompt = "Human: " +  data['question'] + "\nAssistant: "
    else:
        raise ValueError("prompt dataset key error")
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(self, dataset, strategy) -> None:
        super().__init__()
        self.strategy = strategy

        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data)

            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]

