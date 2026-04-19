import random
from abc import ABC
from typing import List

import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience import (
    Experience,
    make_experience_batch,
    remove_padding_in_sequences,
    split_experience_batch,
)
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions


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
        self.items: List[Experience] = []
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

    def __getitem__(self, idx: int) -> Experience:
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
        sample_lengths = [sample.total_length.item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // args.ds.ring_attn_size // args.ds.tensor_parallel_size
        local_train_batch_size = args.train.batch_size // dp_size
        # Expected num_steps assumes a full buffer, but async + partial_rollout
        # can deliver a short buffer at episode boundaries — clamp to avoid
        # iterating past the end of sample_lengths (empty slice → crash in
        # karmarkar_karp with num_mbs=0).
        expected_num_steps = args.rollout.batch_size * args.rollout.n_samples_per_prompt // args.train.batch_size
        num_steps = min(expected_num_steps, len(sample_lengths) // local_train_batch_size)

        # split by train_batch_size, sync num_microbatches across dp
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            num_microbatches.append(
                get_minimum_num_micro_batch_size(
                    sample_lengths[start:end],
                    args.train.max_tokens_per_gpu,
                    args.ds.ring_attn_size,
                    args.ds.tensor_parallel_size,
                )
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # balance the number of microbatches across steps
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
