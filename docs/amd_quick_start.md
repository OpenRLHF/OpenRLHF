## Getting Started with AMD ROCm

### Overview

This document is a quick-start tutorial for running OpenRLHF on AMD ROCm.

**Current hardware scope**:

- GPU targets: MI300X (gfx942)

### Software Baseline and Launch Container

built image for tutorial and validation:

```bash
docker build -f dockerfile/Dockerfile.rocm -t openrlhf-rocm:validated .
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
> If GPU hangs occur during training, try one of the following workarounds.

**Option 1: Disable NCCL IB and P2P**

Add these environment variables to your training script:

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1   # Disabling P2P helps avoid GPU hangs on ROCm.
```

**Option 2: Initialize models sequentially**

In [`train_ppo_ray.py`](../openrlhf/cli/train_ppo_ray.py), replace the parallel model initialization block:

```python
refs = []
refs.extend(
    actor_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path, max_steps, vllm_engines)
)
if ref_model is not None:
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path))
if reward_model is not None and args.reward.model_name_or_path:
    refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward.model_name_or_path))
ray.get(refs)

if critic_model is not None and args.critic.model_name_or_path:
    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    refs = critic_model.async_init_model_from_pretrained(strategy, args.critic.model_name_or_path, max_steps)
    ray.get(refs)
```

with sequential initialization:

```python
actor_refs = actor_model.async_init_model_from_pretrained(
    strategy, args.actor.model_name_or_path, max_steps, vllm_engines
)
ray.get(actor_refs)

if ref_model is not None:
    ref_refs = ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path)
    ray.get(ref_refs)

if reward_model is not None and args.reward.model_name_or_path:
    reward_refs = reward_model.async_init_model_from_pretrained(strategy, args.reward.model_name_or_path)
    ray.get(reward_refs)

if critic_model is not None and args.critic.model_name_or_path:
    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    critic_refs = critic_model.async_init_model_from_pretrained(
        strategy, args.critic.model_name_or_path, max_steps
    )
    ray.get(critic_refs)
```

### Training Scripts

ROCm training scripts live in [`examples/amd_scripts`](../examples/amd_scripts). They mirror the CUDA scripts in [`examples/scripts`](../examples/scripts) with AMD-specific tuning (NCCL settings, vLLM engine layout, and related adjustments).