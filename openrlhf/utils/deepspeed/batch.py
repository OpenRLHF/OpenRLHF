def calculate_gradient_accumulation_steps(
    train_batch_size: int,
    micro_train_batch_size: int,
    world_size: int,
    ring_attn_size: int = 1,
    tensor_parallel_size: int = 1,
    dynamic_batch: bool = False,
) -> int:
    """Validate batch divisibility and return DeepSpeed accumulation steps."""
    parallel_size = ring_attn_size * tensor_parallel_size
    if min(train_batch_size, micro_train_batch_size, world_size, parallel_size) <= 0:
        raise ValueError("Batch sizes, world size, and parallel sizes must be positive")
    if world_size % parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ring_attn_size * tensor_parallel_size "
            f"({ring_attn_size} * {tensor_parallel_size} = {parallel_size})"
        )
    if dynamic_batch:
        return 1

    data_parallel_size = world_size // parallel_size
    batch_quantum = micro_train_batch_size * data_parallel_size
    if train_batch_size % batch_quantum != 0:
        raise ValueError(
            f"train.batch_size ({train_batch_size}) must be divisible by train.micro_batch_size "
            f"({micro_train_batch_size}) * data parallel size ({data_parallel_size}) = {batch_quantum}"
        )
    return train_batch_size // batch_quantum
