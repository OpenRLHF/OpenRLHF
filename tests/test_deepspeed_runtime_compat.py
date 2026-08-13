"""Opt-in two-GPU DeepSpeed/PyTorch ABI smoke test.

Run with:
    torchrun --standalone --nproc-per-node=2 -m pytest -q \
        tests/test_deepspeed_runtime_compat.py
"""

import os

import pytest
import torch


@pytest.mark.skipif(
    "LOCAL_RANK" not in os.environ or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires torchrun with two CUDA devices",
)
def test_deepspeed_zero2_forward_backward_step():
    import deepspeed

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = torch.nn.Linear(16, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        },
    )

    inputs = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    loss = engine(inputs).float().square().mean()
    engine.backward(loss)
    engine.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter).all() for parameter in engine.module.parameters())
