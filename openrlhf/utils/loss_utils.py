from typing import Optional

import torch
import torch.distributed as dist


def _get_dp_group(strategy):
    if not dist.is_available() or not dist.is_initialized():
        return None

    device_mesh = getattr(strategy, "ds_device_mesh", None)
    return device_mesh["dp"].get_group() if device_mesh is not None else None


def get_loss_batch_info(
    strategy,
    loss_mask: torch.Tensor,
    *,
    replay_buffer=None,
    step: Optional[int] = None,
    dynamic_batch: bool = False,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
):
    dp_group = _get_dp_group(strategy)
    dp_size = dist.get_world_size(group=dp_group) if dist.is_available() and dist.is_initialized() else 1

    if dynamic_batch and replay_buffer is not None and step is not None:
        batch_num_tokens = replay_buffer.dynamic_batch_num_tokens[step]
        global_batch_size = replay_buffer.dynamic_global_batch_size[step]

    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum().to(loss_mask.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_num_tokens, op=dist.ReduceOp.SUM, group=dp_group)
    else:
        batch_num_tokens = torch.as_tensor(batch_num_tokens, device=loss_mask.device, dtype=torch.float32)

    if global_batch_size is None:
        sample_mask = loss_mask.reshape(loss_mask.shape[0], -1).sum(dim=-1) > 0
        global_batch_size = sample_mask.sum().to(loss_mask.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_batch_size, op=dist.ReduceOp.SUM, group=dp_group)
    else:
        global_batch_size = torch.as_tensor(global_batch_size, device=loss_mask.device, dtype=torch.float32)

    return {
        "dp_size": dp_size,
        "batch_num_tokens": batch_num_tokens,
        "global_batch_size": global_batch_size,
    }


def _optimizer_step_loss_norm(masks, dp_group, dp_size, gas):
    """Loss normalizers for one optimizer step, from the loss masks of all its micro-batches.

    Returns ``batch_num_tokens`` (total response tokens) and ``global_batch_size`` (total
    non-empty sequences), summed across the whole step and all DP ranks, then divided by gas.

    The ``/ gas`` pre-compensates DeepSpeed, which scales every backward loss by ``1 / gas``:
    with ``batch_num_tokens = total_tokens / gas``, ``aggregate_loss`` returns
    ``masked_sum_i / total_tokens * dp_size * gas``, so after DeepSpeed's ``1 / gas`` the gas
    micro-batches sum to one token-mean over the whole step. ``dp_size`` cancels the gradient
    averaging DeepSpeed/DDP does across DP ranks. (slime applies the same idea via Megatron's
    per-token loss normalizer; DeepSpeed has no equivalent, so we sum the tokens here.)
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    num_tokens = torch.zeros((), dtype=torch.float32, device=device)
    num_samples = torch.zeros((), dtype=torch.float32, device=device)
    for mask in masks:
        mask = mask.to(device=device, dtype=torch.float32)
        num_tokens += mask.sum()
        num_samples += (mask.reshape(mask.shape[0], -1).sum(dim=-1) > 0).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM, group=dp_group)
    return {"dp_size": dp_size, "batch_num_tokens": num_tokens / gas, "global_batch_size": num_samples / gas}


def iter_grad_accum_global_norm(data_iter, strategy, accumulated_gradient, mask_fn):
    """Iterate micro-batches as ``(item, loss_batch_info)``, normalizing per optimizer step.

    DeepSpeed runs ``accumulated_gradient`` (gas) micro-batches per optimizer step. We group
    ``data_iter`` into windows of gas micro-batches (feed order matches DeepSpeed's accumulation
    boundaries) and give every micro-batch in a window the same global normalizers, so the loss
    is one mean over the whole step rather than a mean of per-micro-batch means (which differ
    when micro-batch token counts are uneven). ``mask_fn`` maps an item to the loss mask that
    ``aggregate_loss`` reduces over.
    """
    dp_group = _get_dp_group(strategy)
    dp_size = dist.get_world_size(group=dp_group) if dist.is_available() and dist.is_initialized() else 1
    gas = max(int(accumulated_gradient), 1)

    window = []
    for item in data_iter:
        window.append(item)
        if len(window) == gas:
            info = _optimizer_step_loss_norm([mask_fn(it) for it in window], dp_group, dp_size, gas)
            for it in window:
                yield it, info
            window = []
    if window:  # trailing partial step (only when the data size is not a multiple of gas)
        info = _optimizer_step_loss_norm([mask_fn(it) for it in window], dp_group, dp_size, gas)
        for it in window:
            yield it, info
