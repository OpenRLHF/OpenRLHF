# FSDP2 Strategy for OpenRLHF

This module provides an FSDP2-based training strategy as an alternative to DeepSpeed. It implements the same interface as `DeepspeedStrategy` for seamless integration with existing training code.

## Overview

FSDP2 (Fully Sharded Data Parallel v2) is PyTorch's native distributed training solution that provides:
- Automatic model sharding across GPUs
- Mixed precision training
- Gradient accumulation
- Optional CPU offloading

## Usage

### Command Line

To use FSDP2 backend instead of DeepSpeed, add `--backend fsdp2` to your training command:

```bash
# FSDP2 training
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --param_dtype bf16 \
    --gradient_checkpointing

# DeepSpeed training (default)
deepspeed --module openrlhf.cli.train_sft \
    --backend deepspeed \
    --pretrain meta-llama/Meta-Llama-3-8B \
    ...
```

### FSDP2-specific Arguments

- `--fsdp_tensor_parallel_size`: Tensor parallel size (default: 1)
- `--adam_offload`: Enable CPU offloading for optimizer states

## Implementation Details

### Key Components

1. **FSDP2Strategy** (`fsdp2.py`): Main strategy class that mirrors DeepspeedStrategy interface
   - `setup_distributed()`: Initialize distributed training environment
   - `create_optimizer()`: Create optimizer with proper parameter grouping
   - `backward()`: Perform backward pass with gradient accumulation
   - `optimizer_step()`: Step optimizer with gradient clipping
   - `prepare()`: Wrap model with FSDP2

2. **Utilities** (`fsdp2_utils.py`):
   - `get_optimizer_grouped_parameters()`: Parameter grouping for weight decay
   - `get_grad_norm()`: Compute gradient norm across distributed ranks
   - `clip_grad_by_total_norm_()`: Gradient clipping
   - `to_local_if_dtensor()`: Convert DTensor to local tensor
   - `get_llama_tp_plan()`: Tensor parallel plan for LLaMA models

### FSDP2 Wrapping Process

1. **Tensor Parallelism** (if enabled): Apply column/row-wise parallelism to attention and MLP layers
2. **Layer-wise FSDP**: Shard each transformer layer independently
3. **Root FSDP**: Shard the entire model with `reshard_after_forward=False`

### Mixed Precision Policy

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=dtype,       # bf16 or fp16
    reduce_dtype=torch.float32,  # Gradient reduction in fp32
    output_dtype=torch.float32,  # Outputs in fp32
)
```

### Gradient Accumulation

FSDP2 handles gradient accumulation by:
1. Scaling loss by `1/accumulated_gradient` during backward
2. Only stepping optimizer after `accumulated_gradient` micro-batches

## Comparison with DeepSpeed

| Feature | DeepSpeed | FSDP2 |
|---------|-----------|-------|
| Model Sharding | ZeRO Stage 2/3 | Automatic |
| Mixed Precision | DS config | PyTorch native |
| Gradient Clipping | DS handles | Manual via `get_grad_norm` |
| Tensor Parallel | AutoTP | DTensor-based |
| Activation Checkpointing | DS wrapper | PyTorch `checkpoint_wrapper` |

## Testing

```bash
# Inside Docker container
pip install -e /openrlhf

# Quick import test
python test_fsdp2_import.py

# Full training test
./test_fsdp2_training.sh

# Comparison test (DeepSpeed vs FSDP2)
./run_comparison_test.sh
```

## Known Limitations

1. **Scheduler Recreation**: When model is wrapped with FSDP2, optimizer and scheduler are recreated. Some complex schedulers may not be fully restored.

2. **Checkpoint Format**: FSDP2 checkpoints use `torch.distributed.checkpoint` format, which differs from DeepSpeed format.

3. **Ring Attention**: Ring attention support is experimental with FSDP2.

## References

- [PyTorch FSDP2 Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [NeMo-RL FSDP2 Implementation](https://github.com/NVIDIA/NeMo-RL)
- [OpenRLHF DeepSpeed Strategy](../deepspeed/)
