# FSDP2 Strategy for OpenRLHF

This module provides an FSDP2-based training strategy as an alternative to DeepSpeed. It implements the same interface as `DeepspeedStrategy` as a standalone class for seamless integration with existing training code.

## Dual Backend Architecture

OpenRLHF supports both DeepSpeed and FSDP2 backends as standalone implementations with the same interface:

```
DeepspeedStrategy - DeepSpeed ZeRO-based training
FSDP2Strategy     - PyTorch FSDP2-based training
```

Both backends implement the same interface, enabling:
- Easy backend switching via `--backend deepspeed` or `--backend fsdp2`
- Consistent code structure across different distributed strategies
- Unified configuration and checkpoint handling

## Overview

FSDP2 (Fully Sharded Data Parallel v2) is PyTorch's native distributed training solution that provides:
- Automatic model sharding across GPUs
- Mixed precision training
- Gradient accumulation
- Optional CPU offloading
- **AutoTP (Tensor Parallelism)** using HuggingFace's built-in `._tp_plan`
- **Ring Attention** for sequence parallelism across GPUs

## Usage

### Command Line

To use FSDP2 backend instead of DeepSpeed, add `--backend fsdp2` to your training command:

```bash
# FSDP2 training (data parallelism only)
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

# FSDP2 + AutoTP (tensor parallelism)
# Example: 8 GPUs with TP=2 means DP=4 (4 data parallel groups, each with 2 TP ranks)
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --param_dtype bf16 \
    --gradient_checkpointing \
    --fsdp_tensor_parallel_size 2 \
    --use_hf_tp_plan

# DeepSpeed training (default)
deepspeed --module openrlhf.cli.train_sft \
    --backend deepspeed \
    --pretrain meta-llama/Meta-Llama-3-8B \
    ...
```

### FSDP2-specific Arguments

- `--fsdp_tensor_parallel_size`: Tensor parallel size (default: 1). Set > 1 to enable tensor parallelism.
- `--use_hf_tp_plan`: Use HuggingFace's built-in tensor parallel plan (`._tp_plan`). Requires transformers >= 4.51.
- `--sequence_parallel`: Enable sequence parallelism (requires tensor_parallel_size > 1).
- `--adam_offload`: Enable CPU offloading for optimizer states.

## AutoTP (Tensor Parallelism)

AutoTP enables tensor parallelism using HuggingFace's built-in `._tp_plan` attribute. This is based on NeMo-RL's implementation.

### How It Works

1. **HuggingFace models** (transformers >= 4.51) have built-in tensor parallelism plans via `model._tp_plan`.
2. The `get_hf_tp_plan()` function retrieves TP strategies from:
   - Model class (`model_cls._tp_plan`)
   - Model instance (`model._tp_plan`)
   - Inner model (`model.model._tp_plan`)
3. Special handling for `embed_tokens` and `lm_head` for speedup.
4. String-based parallel styles are translated to DTensor parallelization strategies.

### TP Plan Fallback Priority

When tensor parallelism is enabled (`--fsdp_tensor_parallel_size > 1`), the TP plan is selected in this order:

1. **HuggingFace built-in TP plan** (if `--use_hf_tp_plan` is set and the model supports it)
2. **Optimized TP plan** for known model architectures (LLaMA, Qwen, Gemma)
3. **Default LLaMA-style TP plan** (works for most transformer models)

### Supported Models

- **LLaMA/Llama-2/Llama-3**: Full support with optimized TP plan
- **Qwen2/Qwen3**: Full support with optimized TP plan
- **Gemma3**: Full support with optimized TP plan
- **Any model with `._tp_plan`**: Supported via HuggingFace's built-in plan

### Configuration Requirements

- `world_size` must equal `dp_size * tp_size` (e.g., 8 GPUs = DP4 * TP2)
- `num_attention_heads` must be divisible by `tensor_parallel_size`
- `num_key_value_heads` must be divisible by `tensor_parallel_size`

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
| Tensor Parallel | AutoTP | DTensor-based + HF `._tp_plan` |
| Sequence Parallel | Via ring attention | Via DTensor SequenceParallel |
| Activation Checkpointing | DS wrapper | PyTorch `checkpoint_wrapper` |

## Testing

```bash
# Inside Docker container
pip install -e /openrlhf

# Quick import test
python test_fsdp2_import.py

# Full training test (FSDP2 only)
./examples/scripts/test_sft_fsdp2.sh

# Comparison test (DeepSpeed vs FSDP2)
./run_comparison_test.sh

# FSDP2 + AutoTP test (tensor parallelism)
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 64 \
    --micro_train_batch_size 2 \
    --max_samples 200 \
    --max_epochs 1 \
    --param_dtype bf16 \
    --gradient_checkpointing \
    --fsdp_tensor_parallel_size 2 \
    --use_hf_tp_plan
```

## Known Limitations

1. **Scheduler Recreation**: When model is wrapped with FSDP2, optimizer and scheduler are recreated. Some complex schedulers may not be fully restored.

2. **Checkpoint Format**: FSDP2 checkpoints use `torch.distributed.checkpoint` format, which differs from DeepSpeed format.

3. **Ring Attention**: Ring attention support is experimental with FSDP2.

## References

- [PyTorch FSDP2 Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [NeMo-RL FSDP2 Implementation](https://github.com/NVIDIA/NeMo-RL)
- [OpenRLHF DeepSpeed Strategy](../deepspeed/)
