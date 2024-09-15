import torch
import torch.distributed as dist
from ring_flash_attn import substitute_hf_flash_attn

from openrlhf.utils.distributed_sampler import DistributedSampler


def register_ring_attn(args):
    """
    Create ring attention group and substitute flash attn with ring flash attn.
    """
    if args.ring_attn_size == 1:
        return None

    for i in range(dist.get_world_size() // args.ring_attn_size):
        ring_attn_ranks = list(
            range(
                i * args.ring_attn_size,
                (i + 1) * args.ring_attn_size,
            )
        )
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")
        if dist.get_rank() in ring_attn_ranks:
            ring_attn_group = group

    substitute_hf_flash_attn(ring_attn_group, args.ring_head_stride)
    return ring_attn_group


def get_sampler(args, dataset, consumed_samples=0):
    """
    The rank in the same ring group will share the same sampler.
    """
    if args.ring_attn_size == 1:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() // args.ring_attn_size,
        rank=dist.get_rank() // args.ring_attn_size,
        shuffle=True,
        seed=args.seed,
        consumed_samples=consumed_samples,
    )


def reset_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids
