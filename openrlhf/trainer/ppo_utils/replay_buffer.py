import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict
from itertools import islice

import torch
import torch.nn.functional as F

from openrlhf.models.utils import masked_mean

from .experience_maker import Experience


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
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    base_log_probs: Optional[torch.Tensor] = None
    prompts: Optional[str] = None
    info: Optional[dict] = None


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "base_log_probs",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    prompts = getattr(experience, "prompts")
    if prompts is not None:
        for i, prompt in enumerate(prompts):
            batch_kwargs[i]["prompts"] = prompt

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "base_log_probs",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, value, ret, adv, base_log_prob, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.base_log_probs,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.base_log_probs,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            value[:right_pad] if value is not None else value,
            ret[:right_pad] if ret is not None else ret,
            adv[:right_pad] if value is not None else value,
            base_log_prob[:right_pad] if base_log_prob is not None else base_log_prob,
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
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
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)


class GRPOReplayBuffer(NaiveReplayBuffer):
    """GRPO replay buffer class. Using dictionary to store BufferItem with same query.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts_dic: Dict[str, List[BufferItem]] = defaultdict(list)

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        for item in items:
            self.prompts_dic[item.prompts].append(item)
        if self.limit > 0:
            samples_to_remove = len(self.prompts_dic) - self.limit
            if samples_to_remove > 0:
                keys_to_delete = islice(self.prompts_dic.keys(), samples_to_remove)
                for key in keys_to_delete:
                    del self.prompts_dic[key]

    def clear(self) -> None:
        self.items.clear()
        self.prompts_dic.clear()

    def dic2items(self):
        for _, item_list in self.prompts_dic.items():
            self.items.extend(item_list)

    @torch.no_grad()
    def get_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-8)
        return advantages

    @torch.no_grad()
    def calculate_advantages(self):
        for prompt, item_list in self.prompts_dic.items():
            rewards = [item.info["reward"]  for item in item_list]
            advantages = self.get_advantages(torch.tensor(rewards))
            advantages = [adv.item() for adv in torch.unbind(advantages)]
            for i, adv in enumerate(advantages):
                self.prompts_dic[prompt][i].advantages = torch.tensor([adv])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self.items:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), len(items)], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2)).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)