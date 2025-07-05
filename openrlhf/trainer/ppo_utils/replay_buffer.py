import random
from abc import ABC
from dataclasses import dataclass, fields
from typing import List, Optional, Int


import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.utils import zero_pad_sequences
from openrlhf.utils.seqlen_balancing import  get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    """Split a batch of experiences into individual BufferItems."""
    batch_size = len(experience.sequences)
    # Get fields from BufferItem, excluding 'info'
    keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")
    experience.index = None  # TODO
    # Validate batch size for all attributes
    for key in keys:
        value = getattr(experience, key)
        if value is not None:
            if isinstance(value, (torch.Tensor, list)):
                if len(value) != batch_size:
                    raise ValueError(f"Size of {key} ({len(value)}) does not match batch_size ({batch_size})")

    items = []
    for i in range(batch_size):
        # Process main attributes
        item = {key: (getattr(experience, key)[i] if getattr(experience, key) is not None else None) for key in keys}

        # Process info dictionary
        item["info"] = {}
        for k, v in experience.info.items():
            if isinstance(v, (torch.Tensor, list)):
                if len(v) != batch_size:
                    raise ValueError(f"Size of info[{k}] ({len(v)}) does not match batch_size ({batch_size})")
                item["info"][k] = v[i]
            else:
                raise TypeError(f"Unsupported type for info[{k}]: {type(v)}")

        items.append(BufferItem(**item))

    return items


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    """Combine individual BufferItems into a batch of experiences."""
    if not items:
        raise ValueError("Empty items list")

    # Get fields from BufferItem, excluding 'info'
    keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")

    # Process main attributes
    kwargs = {
        key: (
            zero_pad_sequences([getattr(item, key) for item in items], "right", stack=True)
            if getattr(items[0], key) is not None
            else None
        )
        for key in keys
    }

    # Process info dictionary
    kwargs["info"] = {}
    for key in items[0].info.keys():
        values = [item.info[key] for item in items]
        if not values:
            continue

        # Validate all items have the same type
        first_type = type(values[0])
        if not all(isinstance(v, first_type) for v in values):
            raise TypeError(f"Inconsistent types in info[{key}]")

        # Convert to tensor if all values are numeric
        if all(isinstance(v, (int, float)) for v in values):
            kwargs["info"][key] = torch.tensor(values)
        else:
            kwargs["info"][key] = values

    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        # Calculate right padding using attention_mask
        right_pad = item.attention_mask.flip(0).argmax()
        right_pad = None if right_pad == 0 else -right_pad

        # Remove right padding for all tensors
        for key in [
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
        ]:
            value = getattr(item, key)
            if value is not None:
                setattr(item, key, value[:right_pad])

    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False, dynamic_batch: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.dynamic_batch = dynamic_batch
        self.dynamic_indices: List[List[Int]] = []
        self.num_microbatches: List[int] = []
        self.local_train_batch_size: int = 0

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
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
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        if self.dynamic_batch:
            return len(self.dynamic_indices)
        else:
            return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        if self.dynamic_batch:
            batch = batch[0]
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def set_dynamic_batch(self, strategy):
        args = strategy.args

        world_size = dist.get_world_size()  # TODO, DP
        local_train_batch_size = args.train_batch_size // world_size
        num_steps = args.rollout_batch_size * args.n_samples_per_prompt // args.train_batch_size  # todo, ring、tp
        sample_lengths = [sample.info['total_length'].item() for sample in self.items]

        # split by train_batch_size
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            num_microbatches.append(get_minimum_num_micro_batch_size(sample_lengths[start:end], args.max_tokens_per_gpu))  # [5, 3, 7, 2, 6] 10 -> len([10, 7, 6])=3

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op='max')
        num_microbatches = num_microbatches.tolist()  # [1, 1], return ?

        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)  # List[List[int]], index
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1
        self.num_microbatches = num_microbatches
        self.local_train_batch_size = local_train_batch_size