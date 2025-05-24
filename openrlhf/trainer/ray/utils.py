import os
from typing import List, Optional

import torch
import torch.distributed


# Address https://github.com/ray-project/ray/issues/51117
# This function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
def get_bundle_indices(placement_group, index, length):
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def dynamic_split_batches(attention_mask: torch.Tensor, max_tokens: int, device: torch.device) -> List[List[int]]:
    """Split samples into batches based on token counts and sync across ranks.

    Args:
        attention_mask: Attention mask tensor of shape [batch_size, seq_len]
        max_tokens: Maximum tokens per batch
        device: Device to process on

    Returns:
        List of batch indices, where each batch's total tokens <= max_tokens
    """
    # Calculate token counts and split batches
    token_counts = attention_mask.sum(dim=1)
    batches = []
    current_batch = []
    current_tokens = 0

    for i, count in enumerate(token_counts):
        if current_tokens + count > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [i]
            current_tokens = count
        else:
            current_batch.append(i)
            current_tokens += count

    if current_batch:
        batches.append(current_batch)

    # Sync batch counts across ranks
    if torch.distributed.is_initialized():
        batch_counts = torch.tensor([len(batches)], device=device)
        torch.distributed.all_reduce(batch_counts, op=torch.distributed.ReduceOp.MAX)
        max_batches = batch_counts.item()

        # Pad batches if necessary
        while len(batches) < max_batches:
            batches.append([0])  # Use first sample as padding

        torch.distributed.barrier()

    return batches


def unpad_dynamic_batches(
    results: torch.Tensor, original_length: int, batch_indices: Optional[List[List[int]]] = None
) -> torch.Tensor:
    """Remove padding from results of dynamic batch processing.

    Args:
        results: Combined results from all batches
        original_length: Original number of samples before splitting
        batch_indices: List of batch indices used for splitting

    Returns:
        Tensor with padding removed and original order restored
    """
    if batch_indices is None:
        # If no batch_indices provided, just remove padding
        return results[:original_length]

    # Create a tensor to store results in original order
    ordered_results = torch.zeros_like(results[:original_length])

    # Restore original order using batch_indices
    current_idx = 0
    for batch in batch_indices:
        for idx in batch:
            if idx < original_length:  # Skip padding indices
                ordered_results[idx] = results[current_idx]
            current_idx += 1

    return ordered_results
