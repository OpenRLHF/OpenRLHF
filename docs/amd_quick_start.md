## Getting Started with AMD ROCm

### Overview

This document is a quick-start tutorial for running OpenRLHF on AMD ROCm.

**Current hardware scope**:

- GPU targets: MI300X (gfx942)

### Software Baseline and Launch Container

built image for tutorial and validation:

```bash
docker build -f dockerfile/Dockerfile.AMD -t openrlhf-rocm:validated .
```

**Launch container**

```bash
bash examples/amd_scripts/docker_run.sh
```

### Environment Check (Inside Container)

```bash
# ROCm and visible GPU targets
rocminfo | grep -E "gfx942" || true

# PyTorch + ROCm sanity check
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("rocm :", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("device_0:", torch.cuda.get_device_name(0))
PY
```


### GPU Hang Workarounds

> [!CAUTION]
> If GPU hangs occur during training, add these environment variables to your training script.

**Disable NCCL IB and P2P**

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1   # Disabling P2P helps avoid GPU hangs on ROCm.
```

### Training Scripts

ROCm training scripts live in [`examples/amd_scripts`](../examples/amd_scripts). They mirror the CUDA scripts in [`examples/scripts`](../examples/scripts) with AMD-specific tuning (NCCL settings, vLLM engine layout, and related adjustments).