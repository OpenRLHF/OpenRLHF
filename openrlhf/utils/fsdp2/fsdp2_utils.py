"""FSDP2 utility functions for OpenRLHF.

This module provides utility functions for FSDP2 training, including:
- Optimizer grouped parameters
- Gradient clipping utilities  
- DTensor utilities
- Model parallelization plans
"""

from typing import Union, List, Optional
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    ParallelStyle,
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
    grads_for_norm = [
        to_local_if_dtensor(p.grad.detach()) for p in parameters if p.grad is not None
    ]

    norm_type = float(norm_type)
    total_norm = 0.0

    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm) if grads_for_norm else 0.0
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=dtype, device="cuda")
        # Reduce max across all data-parallel GPUs
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group
        )
        if tp_group is not None:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
        total_norm = float(total_norm_cuda[0].item())
    else:
        total_norm_cuda = torch.tensor(0.0, dtype=dtype, device="cuda")
        for grad in grads_for_norm:
            grad_norm = torch.linalg.vector_norm(grad, ord=norm_type, dtype=dtype)
            total_norm_cuda += torch.pow(grad_norm, norm_type)

        # Sum across all data-parallel GPUs
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )
        if tp_group is not None:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
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
        grads = [
            to_local_if_dtensor(p.grad.detach())
            for p in parameters
            if p.grad is not None
        ]
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
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        base_model_sp_plan = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return base_model_tp_plan
