# FSDP2 Implementation Summary for OpenRLHF

## Overview

This document summarizes the distributed training backend implementation for OpenRLHF that supports both DeepSpeed and FSDP2 backends as standalone, independent implementations with a consistent API interface.

## Implementation Status

### Completed Features

1. **Dual Backend Support** ✅
   - `DeepspeedStrategy` class in `openrlhf/utils/deepspeed/deepspeed.py`
   - `FSDP2Strategy` class in `openrlhf/utils/fsdp2/fsdp2.py`
   - `get_strategy()` function in `openrlhf/utils/utils.py` selects between backends
   - Both backends implement the same interface as standalone classes (no shared base)
   - Backend selection via `--backend deepspeed|fsdp2` CLI argument

2. **FSDP2 Backend** ✅
   - Full implementation in `openrlhf/utils/fsdp2/`
   - Device mesh setup with (dp, sp, tp) dimensions
   - Mixed precision training (bf16/fp16)
   - Gradient accumulation with proper scaling
   - Gradient clipping across distributed ranks
   - Model saving/loading via torch.distributed.checkpoint

3. **AutoTP (Tensor Parallelism)** ✅
   - HuggingFace's built-in `._tp_plan` support via `get_hf_tp_plan()`
   - Optimized TP plans for LLaMA, Qwen, Gemma models
   - Default LLaMA-style TP plan as fallback
   - Proper DeviceMesh setup for (dp, tp) dimensions
   - Validation for attention head divisibility

4. **Ring Attention** ✅
   - Integration with `ring_flash_attn` package
   - Support for sequence parallelism across GPUs
   - Proper tensor slicing and gathering for distributed sequences

5. **CLI Support** ✅
   - `--backend deepspeed|fsdp2` - Backend selection
   - `--fsdp_tensor_parallel_size` - Tensor parallel size
   - `--use_hf_tp_plan` - Use HuggingFace's built-in TP plan
   - `--sequence_parallel` - Enable sequence parallelism
   - `--ring_attn_size` - Ring attention group size

## File Structure

```
openrlhf/
├── utils/
│   ├── __init__.py
│   ├── utils.py                    # get_strategy() for backend selection
│   ├── deepspeed/
│   │   ├── __init__.py
│   │   ├── deepspeed.py           # DeepspeedStrategy class (standalone)
│   │   └── deepspeed_utils.py     # DS utilities
│   └── fsdp2/
│       ├── __init__.py            # Module exports
│       ├── fsdp2.py               # FSDP2Strategy class (standalone)
│       ├── fsdp2_utils.py         # Utilities (TP plans, gradient utils)
│       └── README.md              # Documentation
├── cli/
│   └── train_sft.py               # Updated with FSDP2 arguments
└── models/
    ├── actor.py                   # Backend-aware model wrapper
    └── ring_attn_utils.py         # Ring attention utilities

test/
├── README.md                      # Test documentation
├── run_all_tests.sh               # Run all tests
├── test_sft_deepspeed.sh          # DeepSpeed baseline
├── test_sft_fsdp2.sh              # FSDP2 only
├── test_sft_fsdp2_tp.sh           # FSDP2 + Tensor Parallelism
├── test_sft_fsdp2_ring.sh         # FSDP2 + Ring Attention
├── test_sft_fsdp2_tp_ring.sh      # FSDP2 + TP + Ring (combined)
└── compare_backends.py            # Loss comparison script

examples/scripts/
├── test_sft_deepspeed.sh
├── test_sft_fsdp2.sh
├── test_sft_fsdp2_tp.sh
└── compare_backends.sh
```

## Usage Examples

### Basic FSDP2 Training
```bash
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
```

### FSDP2 + Tensor Parallelism
```bash
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --fsdp_tensor_parallel_size 2 \
    --use_hf_tp_plan \
    --pretrain meta-llama/Meta-Llama-3-8B \
    ...
```

### FSDP2 + Ring Attention
```bash
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --ring_attn_size 2 \
    --packing_samples \
    --max_len 4096 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    ...
```

### FSDP2 + TP + Ring Attention (Combined)
```bash
# 8 GPUs: DP=2, TP=2, Ring=2
torchrun --nproc_per_node=8 -m openrlhf.cli.train_sft \
    --backend fsdp2 \
    --fsdp_tensor_parallel_size 2 \
    --use_hf_tp_plan \
    --ring_attn_size 2 \
    --packing_samples \
    --max_len 2048 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    ...
```

## Key Components

### Common Interface (both backends implement)
Both `DeepspeedStrategy` and `FSDP2Strategy` implement the same interface:
- `setup_distributed()` - Initialize distributed environment
- `create_optimizer()` - Create optimizer with parameter grouping
- `backward()` - Gradient computation with accumulation
- `optimizer_step()` - Optimizer step with gradient clipping
- `prepare()` - Wrap model for distributed training
- `setup_dataloader()` - Set up distributed data loading
- `save_model()` / `load_model()` - Model checkpointing
- `save_ckpt()` / `load_ckpt()` - Training resumption checkpoints
- `all_reduce()` / `all_gather()` - Communication primitives
- `setup_ring_attn()` - Ring attention setup

### FSDP2Strategy Class
- `setup_distributed()` - Initialize distributed environment with device mesh
- `create_optimizer()` - Create optimizer with proper parameter grouping
- `backward()` - Gradient computation with accumulation scaling
- `optimizer_step()` - Optimizer step with gradient clipping
- `prepare()` - Wrap model with FSDP2 and tensor parallelism
- `save_model()` / `load_model()` - HuggingFace-compatible checkpointing
- `save_ckpt()` / `load_ckpt()` - Training resumption checkpoints

### Tensor Parallel Plans
- `get_hf_tp_plan()` - Extract HuggingFace's built-in `._tp_plan`
- `get_optimized_tp_plan()` - Model-specific optimized plans
- `get_llama_tp_plan()` - Default LLaMA-style plan
- `get_qwen_tp_plan()` - Qwen2/Qwen3 plan
- `get_gemma_tp_plan()` - Gemma3 plan

### Gradient Utilities
- `get_grad_norm()` - Compute gradient norm across DP/TP groups
- `clip_grad_by_total_norm_()` - Gradient clipping in-place
- `to_local_if_dtensor()` - Convert DTensor to local tensor

## Validation Checks

The FSDP2 strategy includes validation for:
1. `world_size` divisibility by `ring_attn_size * tensor_parallel_size`
2. Positive data parallel size
3. Positive gradient accumulation steps
4. Attention heads divisibility by tensor parallel size

## Testing

Run all tests:
```bash
cd test/
./run_all_tests.sh
```

Compare backends:
```bash
python compare_backends.py \
    --deepspeed_log logs/deepspeed_test.log \
    --fsdp2_log logs/fsdp2_test.log \
    --threshold 0.05
```

## Success Criteria

- [x] Unified API interface works for both DeepSpeed and FSDP2
- [x] FSDP2-only training implementation complete
- [x] FSDP2 + AutoTP implementation complete with tp_size >= 2
- [x] Ring Attention implementation complete with ring_flash_attn_size >= 2
- [x] Code structure is clean with proper abstraction
- [x] DeepSpeed ring attention support added (setup_ring_attn method)
- [ ] All configurations produce aligned loss values (requires Docker testing)

## Implementation Notes

### Key Changes Made
1. **FSDP2Strategy** - Standalone implementation with TP, Ring Attention support
2. **DeepspeedStrategy** - Standalone implementation with ring attention support
3. **Both backends implement the same interface** - No shared base class for simplicity
4. **Actor model** - Backend-aware, supports both DeepSpeed and FSDP2
5. **CLI** - Backend selection via `--backend deepspeed|fsdp2`

### Test Scripts Available
- `tests/validate_implementation.py` - Validates implementation structure (no GPU required)
- `tests/test_fsdp2_import.py` - Tests module imports
- `tests/test_sft_*.sh` - Training tests for various configurations

## Known Limitations

1. **Scheduler Recreation**: Complex schedulers may not fully restore after FSDP2 wrapping. LambdaLR schedulers (used by transformers) are properly handled by preserving lr_lambda functions.

2. **Checkpoint Format**: FSDP2 checkpoints use `torch.distributed.checkpoint` format, which differs from DeepSpeed format.

3. **Ring Attention**: Ring attention support requires the `ring_flash_attn` package and `--packing_samples` flag.

## References

- [PyTorch FSDP2 Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [NeMo-RL FSDP2 Implementation](https://github.com/NVIDIA/NeMo-RL)
- [HuggingFace Tensor Parallelism](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism)
