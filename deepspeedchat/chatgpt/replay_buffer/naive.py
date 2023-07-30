import random
from typing import List

import torch
from chatgpt.experience_maker.base import Experience
from chatgpt.models.utils import masked_mean
from chatgpt.trainer.strategies import DDPStrategy

from .base import ReplayBuffer
from .utils import BufferItem, make_experience_batch, split_experience_batch, remove_padding_in_sequences


class NaiveReplayBuffer(ReplayBuffer):
    """Naive replay buffer class. It stores experience.

     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
         cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True) -> None:
        super().__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device('cpu'))
        items = split_experience_batch(experience)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == 'advantages'
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()
        action_masks_vector = torch.cat(action_masks).flatten()
        
        # For DDP
        if isinstance(strategy, DDPStrategy):
            # mean
            sum_and_count = torch.tensor([items_vector.sum(), action_masks_vector.sum()], 
                                         device=items_vector.device)
            all_sum, all_count = strategy.all_reduce(sum_and_count, 'sum')
            mean = all_sum / all_count
            # std
            std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
            all_std = strategy.all_reduce(std, 'sum')
            rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
        else:
            mean = masked_mean(items_vector, action_masks_vector, dim=0)
            rstd = masked_mean((items_vector - mean).pow(2), action_masks_vector, dim=0).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
