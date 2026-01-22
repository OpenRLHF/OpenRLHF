"""FSDP2 utility functions for OpenRLHF.

This module provides utility functions for FSDP2 training, including:
- Optimizer grouped parameters
- Gradient clipping utilities
- DTensor utilities
- Model parallelization plans
- HuggingFace tp_plan support (AutoTP)
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard


def get_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_name_list: List[str] = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    """Get optimizer grouped parameters with weight decay handling.

    Args:
        model: The model to get parameters from
        weight_decay: Weight decay value for parameters not in no_decay_name_list
        no_decay_name_list: List of parameter name patterns to exclude from weight decay

    Returns:
        List of parameter groups with appropriate weight decay
    """
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor.

    Args:
        tensor: A torch.Tensor or DTensor

    Returns:
        The local tensor (unwrapped if DTensor)
    """
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def get_grad_norm(
    parameters: Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_group: torch.distributed.ProcessGroup,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Args:
        parameters: Parameters to compute gradient norm for
        dp_group: Process group for data parallel communication
        tp_group: Process group for tensor parallel communication (optional)
        norm_type: Type of the used p-norm
        dtype: Data type for norm computation

    Returns:
        Total norm of the gradients
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Get gradients
    grads_for_norm = [to_local_if_dtensor(p.grad.detach()) for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    total_norm = 0.0

    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm) if grads_for_norm else 0.0
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=dtype, device="cuda")
        # Reduce max across all data-parallel GPUs
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        if tp_group is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        total_norm = float(total_norm_cuda[0].item())
    else:
        total_norm_cuda = torch.tensor(0.0, dtype=dtype, device="cuda")
        for grad in grads_for_norm:
            grad_norm = torch.linalg.vector_norm(grad, ord=norm_type, dtype=dtype)
            total_norm_cuda += torch.pow(grad_norm, norm_type)

        # Sum across all data-parallel GPUs
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        if tp_group is not None:
            torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        total_norm = float(total_norm_cuda.item() ** (1.0 / norm_type))

    return total_norm


def clip_grad_by_total_norm_(
    parameters: Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_grad_norm: Union[int, float],
    total_norm: float,
):
    """Clips gradient of an iterable of parameters by total norm.

    Note that the gradients are modified in place.

    Args:
        parameters: Parameters to clip gradients for
        max_grad_norm: Maximum norm of the gradients
        total_norm: The pre-computed total norm of the gradients to use for scaling
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Scale coefficient
    clip_coeff = max_grad_norm / (total_norm + 1.0e-6)

    if clip_coeff < 1.0:
        grads = [to_local_if_dtensor(p.grad.detach()) for p in parameters if p.grad is not None]
        for g in grads:
            g.mul_(clip_coeff)


def get_llama_tp_plan(sequence_parallel: bool = False) -> dict:
    """Get tensor parallel plan for LLaMA-style models.

    Args:
        sequence_parallel: Whether to enable sequence parallelism

    Returns:
        Dictionary mapping module paths to parallel styles
    """
    base_model_tp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        # Use Replicate() output and use_local_output=True to gather the lm_head output
        # This ensures compatibility with loss computation which expects a regular tensor
        "lm_head": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
    }

    if sequence_parallel:
        base_model_sp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate(), use_local_output=True),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return base_model_tp_plan


@lru_cache
def translate_parallel_style(style: str) -> ParallelStyle:
    """Translate parallel style string to ParallelStyle object.

    This function translates HuggingFace's string-based parallel style
    specifications to PyTorch DTensor parallelization strategies.

    Based on NeMo-RL implementation:
    https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/models/dtensor/parallelize.py

    Args:
        style: String representation of parallel style.
               Supported values: "colwise", "rowwise", "colwise_rep",
               "rowwise_rep", "sequence_parallel"

    Returns:
        Corresponding ParallelStyle object

    Raises:
        ValueError: If the style is not recognized
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def get_hf_tp_plan(model: nn.Module) -> Dict[str, ParallelStyle]:
    """Get tensor parallel plan from HuggingFace model's built-in `._tp_plan`.

    This function retrieves tensor parallelism strategies from HuggingFace models
    that have built-in TP support (transformers >= 4.51). It handles:
    - TP strategies from model class (`model_cls._tp_plan`)
    - TP strategies from model instance (`model._tp_plan`)
    - TP strategies from inner model (`model.model._tp_plan`)
    - Special handling for embed_tokens and lm_head for speedup

    Based on NeMo-RL implementation:
    https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/models/dtensor/parallelize.py

    Args:
        model: A HuggingFace model instance (PreTrainedModel)

    Returns:
        Dictionary mapping model component paths to their parallelization strategies

    Raises:
        AssertionError: If no TP plan is found for the model

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tp_plan = get_hf_tp_plan(model)
        >>> print(tp_plan.keys())
    """
    model_cls = type(model)

    # Determine model structure and prefix
    # Handle different model architectures
    inner_model = model.model if hasattr(model, "model") else model
    model_prefix = "model"
    config = model.config if hasattr(model, "config") else None

    # Handle Vision-Language models with different structures
    model_cls_name = model_cls.__name__

    if "Qwen2VL" in model_cls_name or "Qwen2_5_VL" in model_cls_name:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            inner_model = model.model.language_model
            model_prefix = "model.language_model"
            config = model.model.language_model.config if hasattr(model.model.language_model, "config") else config
    elif "Gemma3ForConditionalGeneration" in model_cls_name:
        if hasattr(model, "language_model"):
            inner_model = model.language_model
            model_prefix = "language_model"
            config = model.config.text_config if hasattr(model.config, "text_config") else config
    elif "Llama4ForConditionalGeneration" in model_cls_name:
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            inner_model = model.language_model.model
            model_prefix = "language_model.model"
            config = model.language_model.model.config if hasattr(model.language_model.model, "config") else config
    elif any(name in model_cls_name for name in ["Llava", "LlavaNext", "LlavaNextVideo", "LlavaOnevision"]):
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            inner_model = model.model.language_model
            model_prefix = "model.language_model"
            config = model.model.language_model.config if hasattr(model.model.language_model, "config") else config
    elif "Mistral3ForConditionalGeneration" in model_cls_name:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            inner_model = model.model.language_model
            model_prefix = "model.language_model"
            config = model.model.language_model.config if hasattr(model.model.language_model, "config") else config

    hf_tp_plan: Dict[str, Any] = {}

    # Helper function to add prefix to keys that don't have it
    def add_prefix_if_needed(plan: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to keys that are relative paths (start with 'layers.')"""
        result = {}
        for k, v in plan.items():
            # Keys like 'layers.*' need the model prefix, but 'lm_head' doesn't
            if k.startswith("layers."):
                result[f"{prefix}.{k}"] = v
            elif k.startswith("embed_tokens"):
                result[f"{prefix}.{k}"] = v
            else:
                # Keys like 'lm_head' or already prefixed keys
                result[k] = v
        return result

    # Collect TP plan from model class (model_cls._tp_plan will override after xxxForCausalLM.post_init())
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        assert isinstance(model_cls._tp_plan, dict), f"model_cls._tp_plan is not a dict: {model_cls._tp_plan}"
        # Class-level tp_plan may have relative paths, add prefix
        prefixed_plan = add_prefix_if_needed(model_cls._tp_plan, model_prefix)
        hf_tp_plan.update(prefixed_plan)

    # Collect TP plan from model instance
    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        # Instance-level tp_plan may also have relative paths
        prefixed_plan = add_prefix_if_needed(model._tp_plan, model_prefix)
        hf_tp_plan.update(prefixed_plan)

    # Collect TP plan from inner model
    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    if len(hf_tp_plan) == 0:
        raise AssertionError(
            f"HuggingFace tp plan is not supported for {model_cls}. "
            f"Please set use_hf_tp_plan=False or provide a custom tensor parallel plan. "
            f"Alternatively, the model may not have built-in TP support (requires transformers >= 4.51)."
        )

    # Add embed_tokens if not present (set to rowwise_rep for speedup)
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    # Convert string-based parallel styles to ParallelStyle objects
    converted_plan: Dict[str, ParallelStyle] = {}
    for k, v in hf_tp_plan.items():
        # For lm_head with colwise parallelism, we need to gather the output for loss computation
        # use_local_output=True gathers the sharded output to all ranks
        if (k == "lm_head" or k.endswith(".lm_head")) and v == "colwise_rep":
            converted_plan[k] = ColwiseParallel(output_layouts=Replicate(), use_local_output=True)
        elif isinstance(v, str):
            converted_plan[k] = translate_parallel_style(v)
        elif isinstance(v, ParallelStyle):
            converted_plan[k] = v
        else:
            raise ValueError(f"Unknown parallel style type for key {k}: {type(v)}")

    return converted_plan


# Model-specific TP plan functions for models without HF built-in support
# These provide optimized plans based on NeMo-RL's implementation


def get_qwen_tp_plan(sequence_parallel: bool = False) -> Dict[str, ParallelStyle]:
    """Get tensor parallel plan for Qwen2/Qwen3 models.

    Args:
        sequence_parallel: Whether to enable sequence parallelism

    Returns:
        Dictionary mapping module paths to parallel styles
    """
    if sequence_parallel:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }
    else:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }

    return base_model_tp_plan


def get_gemma_tp_plan(model_prefix: str = "model", sequence_parallel: bool = False) -> Dict[str, ParallelStyle]:
    """Get tensor parallel plan for Gemma3 models.

    Args:
        model_prefix: Prefix for model path (e.g., "model" or "model.language_model")
        sequence_parallel: Whether to enable sequence parallelism

    Returns:
        Dictionary mapping module paths to parallel styles
    """
    base_model_tp_plan: Dict[str, ParallelStyle] = {
        f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
    }

    if sequence_parallel:
        base_model_sp_plan = {
            f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),
            f"{model_prefix}.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate(), use_local_output=True),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return base_model_tp_plan


# Mapping of model types to their TP plan functions
# This allows automatic selection of the right TP plan based on model architecture
MODEL_TP_PLAN_FUNCTIONS = {
    "LlamaForCausalLM": get_llama_tp_plan,
    "Qwen2ForCausalLM": get_qwen_tp_plan,
    "Qwen3ForCausalLM": get_qwen_tp_plan,
    "Gemma3ForCausalLM": lambda sp=False: get_gemma_tp_plan("model", sp),
    "Gemma3ForConditionalGeneration": lambda sp=False: get_gemma_tp_plan("model.language_model", sp),
}


def _move_fsdp2_optimizer_to_device(optimizer, device: str) -> None:
    """Move optimizer states to specified device.

    Args:
        optimizer: The optimizer whose states should be moved
        device: Target device ("cpu" or "cuda")
    """
    if optimizer is None:
        return

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, (DTensor, torch.Tensor)):
                state[k] = v.to(device)


def _move_fsdp2_buffers_to_device(model: nn.Module, device: str) -> None:
    """Move FSDP2 model buffers to specified device.

    FSDP2 modules may not move buffers automatically, so we need to do this explicitly.

    Args:
        model: The FSDP2-wrapped model
        device: Target device ("cpu" or "cuda")
    """
    for buf in model.buffers():
        if buf.device.type != device:
            torch.utils.swap_tensors(buf, buf.to(device))


def _move_fsdp2_model_to_device(model: nn.Module, device: str) -> nn.Module:
    """Move FSDP2 model to specified device.

    For FSDP2, we need to:
    1. Move the model parameters and buffers to the target device
    2. Handle buffers explicitly since FSDP2 may not move them automatically

    Based on NeMo-RL's implementation:
    https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/models/policy/workers/dtensor_policy_worker.py

    Args:
        model: The FSDP2-wrapped model
        device: Target device ("cpu" or "cuda")

    Returns:
        The model with parameters on the target device
    """
    # Move buffers explicitly - FSDP2 modules may not move buffers automatically
    _move_fsdp2_buffers_to_device(model, device)

    # Move model parameters to target device
    model = model.to(device)
    return model


def offload_fsdp2_optimizer(optimizer, device: str = "cpu") -> None:
    """Offload optimizer states to CPU memory.

    Args:
        optimizer: The optimizer whose states should be offloaded
        device: Target device for offloading (default: "cpu")
    """
    _move_fsdp2_optimizer_to_device(optimizer, device)


def reload_fsdp2_optimizer(optimizer, device: str = "cuda") -> None:
    """Reload optimizer states to GPU memory.

    Args:
        optimizer: The optimizer whose states should be reloaded
        device: Target device for reloading (default: "cuda")
    """
    _move_fsdp2_optimizer_to_device(optimizer, device)


def offload_fsdp2_states(model: nn.Module, optimizer=None) -> None:
    """Offload FSDP2 model and optimizer states to CPU for memory efficiency.

    This is useful when you want to free up GPU memory during inference/generation
    phases in colocate mode. Call reload_fsdp2_states() before training.

    Based on NeMo-RL's offload_after_refit implementation:
    https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/models/policy/workers/dtensor_policy_worker.py

    Args:
        model: The FSDP2-wrapped model
        optimizer: Optional optimizer to offload states
    """
    import gc

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Log GPU memory before offload
    if rank == 0:
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[FSDP2 Offload] GPU memory before: {mem_before:.2f} GB", flush=True)

    # Move model to CPU
    _move_fsdp2_model_to_device(model, "cpu")

    # Set model to eval mode after offloading
    model.eval()

    # Offload optimizer states to CPU
    if optimizer is not None:
        _move_fsdp2_optimizer_to_device(optimizer, "cpu")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.cuda.synchronize()

    # Log GPU memory after offload
    if rank == 0:
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(
            f"[FSDP2 Offload] GPU memory after: {mem_after:.2f} GB (freed: {mem_before - mem_after:.2f} GB)",
            flush=True,
        )


def reload_fsdp2_states(model: nn.Module, optimizer=None, training_mode: bool = True) -> None:
    """Reload FSDP2 model and optimizer states to GPU.

    This should be called before training/inference after offload_fsdp2_states was used.

    Based on NeMo-RL's prepare_for_training implementation:
    https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/models/policy/workers/dtensor_policy_worker.py

    Args:
        model: The FSDP2-wrapped model
        optimizer: Optional optimizer to reload states
        training_mode: Whether to set model to train mode (True) or eval mode (False).
                      Default is True for backward compatibility. Set to False for
                      models like reward models that should stay in eval mode.
    """
    import gc

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Log GPU memory before reload
    if rank == 0:
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[FSDP2 Reload] GPU memory before: {mem_before:.2f} GB", flush=True)

    # Move model back to CUDA
    _move_fsdp2_model_to_device(model, "cuda")

    # Set model to appropriate mode
    if training_mode:
        model.train()
    else:
        model.eval()

    # Reload optimizer states to GPU
    if optimizer is not None:
        _move_fsdp2_optimizer_to_device(optimizer, "cuda")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.cuda.synchronize()

    # Log GPU memory after reload
    if rank == 0:
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(
            f"[FSDP2 Reload] GPU memory after: {mem_after:.2f} GB (used: {mem_after - mem_before:.2f} GB)", flush=True
        )


def gather_fsdp2_params_for_broadcast(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Gather full model parameters from FSDP2 shards for weight synchronization.

    This function collects the full (unsharded) model parameters that can be
    broadcast to vLLM engines for weight synchronization.

    Args:
        model: The FSDP2-wrapped model

    Returns:
        Dictionary mapping parameter names to full (gathered) tensors
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    # Get full state dict - this gathers from all FSDP shards
    with torch.no_grad():
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,  # Keep on GPU for broadcast
            ),
        )

    return state_dict


def get_optimized_tp_plan(model: nn.Module, sequence_parallel: bool = False) -> Optional[Dict[str, ParallelStyle]]:
    """Get optimized tensor parallel plan for specific model architectures.

    This function returns a hand-tuned TP plan for models that have known
    optimal parallelization strategies. Falls back to None if no optimized
    plan is available for the model type.

    Args:
        model: The model to get the TP plan for
        sequence_parallel: Whether to enable sequence parallelism

    Returns:
        Dictionary mapping module paths to parallel styles, or None if no
        optimized plan is available
    """
    model_cls_name = type(model).__name__

    if model_cls_name in MODEL_TP_PLAN_FUNCTIONS:
        return MODEL_TP_PLAN_FUNCTIONS[model_cls_name](sequence_parallel)

    return None
