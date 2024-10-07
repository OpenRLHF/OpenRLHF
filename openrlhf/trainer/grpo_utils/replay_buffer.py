import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .experience_maker import GRPOExperience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    returns: (1)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

def split_experience_batch(experience: GRPOExperience, n_responses: int) -> List[BufferItem]:
    batch_size = experience.sequences.size(0) // n_responses # number of different prompts
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        vals = torch.split(value, n_responses, dim=0)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.split(v, n_responses, dim=0)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.vstack(padded_sequences)

def make_experience_batch(items: List[BufferItem]) -> GRPOExperience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        batch_data = zero_pad_sequences(vals, "left")
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.concatenate([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return GRPOExperience(**kwargs)

def remove_padding_in_sequences(items):
    for item in items:
        seqs, act_log_probs, base_log_probs, advs, att_masks, act_masks = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = min((1 - act_masks[i, :].long()).sum() for i in range(act_masks.shape[0]))
        right_pad = None if right_pad == 0 else -right_pad
        left_pad = min((att_masks[i, :].long().argmax() for i in range(att_masks.shape[0])))
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seqs[:, left_pad:right_pad],
            act_log_probs[:, :right_pad],
            base_log_probs[:, :right_pad],
            advs[:, :right_pad],
            att_masks[:, left_pad:right_pad],
            act_masks[:, :right_pad],
        )
    return items


class GRPOReplayBuffer(ABC):
    """GRPO replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, micro_train_batch_size: int, n_responses: int, limit: int = 0, cpu_offload: bool = True) -> None:
        super().__init__()
        self.sample_batch_size = micro_train_batch_size // n_responses
        self.n_responses = n_responses
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: GRPOExperience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience, self.n_responses)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> GRPOExperience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> GRPOExperience:
        experience = make_experience_batch(batch)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        for item in self:
            items.append(getattr(item, attribute))
        items_vector = torch.cat(items).float().flatten()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), items_vector.shape[0]], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2)).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)

