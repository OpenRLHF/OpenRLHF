# OpenRLHF Backend Test Suite

This directory contains test scripts for comparing DeepSpeed and FSDP2 backends in OpenRLHF.

## Test Scripts

| Script | Description |
|--------|-------------|
| `validate_implementation.py` | **Run first** - Validates FSDP2 implementation structure |
| `test_fsdp2_import.py` | Tests all FSDP2 module imports |
| `test_sft_deepspeed.sh` | DeepSpeed baseline SFT training |
| `test_sft_fsdp2.sh` | FSDP2 only (data parallelism) SFT training |
| `test_sft_fsdp2_tp.sh` | FSDP2 + Tensor Parallelism (AutoTP) SFT training |
| `test_sft_fsdp2_ring.sh` | FSDP2 + Ring Attention SFT training |
| `test_sft_fsdp2_tp_ring.sh` | FSDP2 + TP + Ring (combined) SFT training |
| `compare_backends.py` | Python script to compare loss values |
| `run_all_tests.sh` | Run all tests and compare results |

## Quick Start

### Validate Implementation

Before running training tests, validate the implementation:

```bash
# Run validation script (no GPU required)
python validate_implementation.py

# Run import tests
python test_fsdp2_import.py
```

### Run Individual Tests

```bash
# DeepSpeed baseline
./test_sft_deepspeed.sh

# FSDP2 only
./test_sft_fsdp2.sh

# FSDP2 + AutoTP (Tensor Parallelism)
./test_sft_fsdp2_tp.sh

# FSDP2 + Ring Attention
./test_sft_fsdp2_ring.sh

# FSDP2 + AutoTP + Ring Attention (combined)
./test_sft_fsdp2_tp_ring.sh
```

### Run All Tests

```bash
# Run all tests and compare
./run_all_tests.sh

# Skip specific tests
SKIP_DEEPSPEED=1 ./run_all_tests.sh
SKIP_RING=1 ./run_all_tests.sh
```

### Compare Backends

```bash
# Compare loss values from log files
python compare_backends.py \
    --deepspeed_log logs/deepspeed_test.log \
    --fsdp2_log logs/fsdp2_test.log \
    --threshold 0.05 \
    --plot logs/comparison.png
```

## Configuration

All test scripts support environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | 8 | Number of GPUs to use |
| `MAX_SAMPLES` | 1000 | Maximum training samples |
| `MODEL` | meta-llama/Meta-Llama-3-8B | Pretrained model |
| `DATASET` | Open-Orca/OpenOrca | Training dataset |
| `OUTPUT_DIR` | ./checkpoint/test_* | Checkpoint output directory |
| `LOG_DIR` | ./logs | Log output directory |
| `TP_SIZE` | 2 | Tensor parallel size (for TP test) |
| `RING_SIZE` | 2 | Ring attention size (for Ring test) |

Example:
```bash
NUM_GPUS=4 MAX_SAMPLES=500 ./test_sft_fsdp2.sh
```

## Backend Configurations

### DeepSpeed (Baseline)
- Backend: `--backend deepspeed`
- ZeRO Stage: 2
- Launcher: `deepspeed --module`

### FSDP2 (Data Parallelism)
- Backend: `--backend fsdp2`
- Pure data parallelism across all GPUs
- Launcher: `torchrun`

### FSDP2 + AutoTP (Tensor Parallelism)
- Backend: `--backend fsdp2`
- `--fsdp_tensor_parallel_size 2` (or higher)
- `--use_hf_tp_plan` for HuggingFace's built-in TP plan
- Effective parallelism: DP = world_size / TP

### FSDP2 + Ring Attention
- Backend: `--backend fsdp2`
- `--ring_attn_size 2` (or higher)
- `--packing_samples` (required for Ring Attention)
- Enables training on longer sequences

## Success Criteria

Tests are considered successful when:
1. Training completes without errors
2. Loss values decrease during training
3. DeepSpeed and FSDP2 loss values are aligned (<5% difference)

## Troubleshooting

### Out of Memory
- Reduce `--micro_train_batch_size`
- Enable `--gradient_checkpointing`
- Use smaller model or fewer samples

### Ring Attention Issues
- Ensure `--packing_samples` is enabled
- Check that `ring_attn_size` divides `NUM_GPUS`
- Install `ring_flash_attn` package

### Tensor Parallelism Issues
- Ensure `num_attention_heads % TP_SIZE == 0`
- Ensure `num_key_value_heads % TP_SIZE == 0`
- Check that `TP_SIZE` divides `NUM_GPUS`

## Output

Logs are saved to `./logs/` directory:
- `deepspeed_test.log` - DeepSpeed training log
- `fsdp2_test.log` - FSDP2 training log
- `fsdp2_tp{N}_test.log` - FSDP2 + TP training log
- `fsdp2_ring{N}_test.log` - FSDP2 + Ring Attention training log
- `fsdp2_tp{N}_ring{M}_test.log` - FSDP2 + TP + Ring Attention combined training log

## Dual Backend Architecture

Both DeepSpeed and FSDP2 backends are standalone implementations with the same interface, providing:
- Consistent API across backends
- Easy backend switching via `--backend` flag
- Unified configuration and checkpoint handling

See `openrlhf/utils/deepspeed/deepspeed.py` and `openrlhf/utils/fsdp2/fsdp2.py` for the implementations.
