import random
from abc import ABC
from dataclasses import dataclass, fields
from typing import List, Optional

import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import zero_pad_sequences


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
    rollout_log_probs: torch.Tensor
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
    experience.index = None

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
        keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")
        for key in keys:
            value = getattr(item, key)
            if value is not None:
                setattr(item, key, value[:right_pad])

    return items


def balance_experiences(experiences, args):
    """
    Balance experience accross dp
    Example:
        sorted lengths: [8,7,6,5,4,3,2,1], effective_num: 2
        first_half: [[8,7], [6,5]], last_half: [[3,4], [1,2]], interval_items: [[8,7], [1,2], [6,5], [3,4]]
        interval_merged: [[8,1,6,3], [7,2,5,4]]
    """
    # split experience, sort by total_length
    items_all = []
    for item in experiences:
        items_all.extend(split_experience_batch(item))
    items_all.sort(key=lambda x: x.info["total_length"], reverse=True)

    # split experience into chunks
    effective_num = (
        args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    )
    split_items = [items_all[i : i + effective_num] for i in range(0, len(items_all), effective_num)]
    half = len(split_items) // 2
    first_half = split_items[:half]
    last_half = [item[::-1] for item in split_items[half:]]

    # balance distribution by intervaling chunks
    interval_items = []
    for i in range(half):
        interval_items.append(first_half[i])
        interval_items.append(last_half[-(i + 1)])
    if len(last_half) > len(first_half):
        interval_items.append(last_half[0])

    interval_merged = list(zip(*interval_items))
    return [make_experience_batch(items) for items in interval_merged]


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
        packing_samples: bool = False,
        dynamic_batch: bool = False,
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
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_loss_scale: List[float] = []
        self.dynamic_optimizer_step: List[int] = []

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
            indices = self.dynamic_indices[idx]
            return [self.items[i] for i in indices]
        else:
            return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch:
            batch = batch[0]
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def setup_dynamic_batch(self, strategy):
        args = strategy.args
        sample_lengths = [sample.info["total_length"].item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        local_train_batch_size = args.train_batch_size // dp_size
        num_steps = args.rollout_batch_size * args.n_samples_per_prompt // args.train_batch_size

        # split by train_batch_size, sync num_microbatches across dp
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            num_microbatches.append(
                get_minimum_num_micro_batch_size(
                    sample_lengths[start:end],
                    args.train_max_tokens_per_gpu,
                    args.ring_attn_size,
                    args.ds_tensor_parallel_size,
                )
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # balance the number of mirobatches across steps
        micro_batch_indices = []
        data_partitions = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)  # List[List[int]], index
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)
        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        # adjust optimizer step and loss scale
        loss_scales = []
        optimizer_steps = []
        for partitions in data_partitions:
            sample_num = sum(len(partition) for partition in partitions)
            loss_scale = [len(partition) / sample_num for partition in partitions]
            optimizer_step = [0] * (len(partitions) - 1) + [1]
            loss_scales.extend(loss_scale)
            optimizer_steps.extend(optimizer_step)
        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
