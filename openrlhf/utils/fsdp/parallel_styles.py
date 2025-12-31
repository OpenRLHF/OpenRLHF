# Copyright (c) 2025, OpenRLHF Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoRA-aware Tensor Parallel styles.

Based on NeMo Automodel implementation, provides LoRA-aware ParallelStyle subclasses
to ensure LoRA adapter layers are correctly handled during TP sharding.

Usage:
    from openrlhf.utils.fsdp.parallel_styles import translate_to_lora
    
    # Convert plan to LoRA-aware version before applying TP
    tp_plan = {k: translate_to_lora(v) for k, v in original_plan.items()}
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_torch_tp_available() -> bool:
    """Check if PyTorch version supports tensor parallel API."""
    try:
        from packaging import version
        torch_version = version.parse(torch.__version__.split("+")[0])
        return torch_version >= version.parse("2.5.0")
    except Exception:
        return False


if _is_torch_tp_available():
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        SequenceParallel,
    )
    from torch.distributed.device_mesh import DeviceMesh


def _distribute_param(
    module: nn.Module,
    name: str,
    device_mesh: "DeviceMesh",
    placements: list,
    src_data_rank: Optional[int] = None,
) -> None:
    """
    Distribute a module's parameter to the specified device mesh.
    
    Args:
        module: Target module
        name: Parameter name
        device_mesh: Device mesh for distribution
        placements: DTensor placement strategy
        src_data_rank: Source data rank (optional)
    """
    param = getattr(module, name)
    dist_param = nn.Parameter(
        distribute_tensor(param, device_mesh, placements, src_data_rank=src_data_rank),
        requires_grad=param.requires_grad,
    )
    module.register_parameter(name, dist_param)


class ColwiseParallelLora(ColwiseParallel):
    """
    LoRA-aware ColwiseParallel.
    
    For standard Linear layers, uses ColwiseParallel's default behavior (weight Shard(0)).
    For LoRA layers (lora_A, lora_B), correctly handles sharding and adds hooks to gather output.
    """
    
    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: "DeviceMesh") -> None:
        """
        Handle Linear layer sharding, including LoRA adapters.
        
        ColwiseParallel shards weight along dim=0 (row-wise split),
        meaning in Linear(input * weight^T), output is sharded along dim=-1.
        """
        def _get_module_and_name(module: nn.Module, param_name: str):
            """Get LoRA submodule and parameter name."""
            if param_name.endswith("lora_A.weight"):
                if hasattr(module, "lora_A"):
                    return module.lora_A, "weight"
                # Some LoRA implementations use different attribute names
                for attr in ["lora_A", "lora_a"]:
                    if hasattr(module, attr):
                        return getattr(module, attr), "weight"
            elif param_name.endswith("lora_B.weight"):
                if hasattr(module, "lora_B"):
                    return module.lora_B, "weight"
                for attr in ["lora_B", "lora_b"]:
                    if hasattr(module, attr):
                        return getattr(module, attr), "weight"
            return module, param_name

        # Apply Shard(0) to all parameters
        for param_name, param in module.named_parameters():
            _module, _name = _get_module_and_name(module, param_name)
            _distribute_param(_module, _name, device_mesh, [Shard(0)], self.src_data_rank)

        # Register forward hook for lora_A to all-gather its output
        # Since lora_A's output is sharded, we need to gather before passing to lora_B
        def lora_a_output_hook(mod, input, output):
            if isinstance(output, DTensor):
                # If output has Shard placement, redistribute to Replicate
                if any(isinstance(p, Shard) for p in output.placements):
                    output = output.redistribute(
                        device_mesh=output.device_mesh,
                        placements=[Replicate()],
                    )
            return output

        # Register the hook
        if hasattr(module, "lora_A"):
            module.lora_A.register_forward_hook(lora_a_output_hook)
        elif hasattr(module, "lora_a"):
            module.lora_a.register_forward_hook(lora_a_output_hook)

    def _partition_embedding_fn(self, name: str, module: nn.Module, device_mesh: "DeviceMesh") -> None:
        """
        Handle Embedding layer sharding.
        
        ColwiseParallel uses Shard(1) for Embedding, i.e., split along embedding_dim.
        """
        for param_name, param in module.named_parameters():
            _distribute_param(module, param_name, device_mesh, [Shard(1)], self.src_data_rank)


class RowwiseParallelLora(RowwiseParallel):
    """
    LoRA-aware RowwiseParallel.
    
    For standard Linear layers, uses RowwiseParallel's default behavior (weight Shard(1)).
    For LoRA layers (lora_A, lora_B), both are sharded along dim=1.
    """
    
    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: "DeviceMesh") -> None:
        """
        Handle Linear layer sharding, including LoRA adapters.
        
        RowwiseParallel shards weight along dim=1 (column-wise split),
        input needs to be sharded along dim=-1, output is complete after all-reduce.
        """
        # weight Shard(1), bias Replicate
        _distribute_param(module, "weight", device_mesh, [Shard(1)], self.src_data_rank)
        if getattr(module, "bias", None) is not None:
            _distribute_param(module, "bias", device_mesh, [Replicate()], self.src_data_rank)
        
        # LoRA layers: both lora_A and lora_B are sharded with Shard(1)
        if hasattr(module, "lora_A"):
            _distribute_param(module.lora_A, "weight", device_mesh, [Shard(1)], self.src_data_rank)
            _distribute_param(module.lora_B, "weight", device_mesh, [Shard(1)], self.src_data_rank)
        elif hasattr(module, "lora_a"):
            _distribute_param(module.lora_a, "weight", device_mesh, [Shard(1)], self.src_data_rank)
            _distribute_param(module.lora_b, "weight", device_mesh, [Shard(1)], self.src_data_rank)

    def _partition_embedding_fn(self, name: str, module: nn.Module, device_mesh: "DeviceMesh") -> None:
        """
        Handle Embedding layer sharding.
        
        RowwiseParallel uses Shard(0) for Embedding, i.e., split along vocab.
        """
        for param_name, param in module.named_parameters():
            _distribute_param(module, param_name, device_mesh, [Shard(0)], self.src_data_rank)


class SequenceParallelLora(SequenceParallel):
    """
    LoRA-aware SequenceParallel.
    
    Used for modules requiring sequence parallel, such as LayerNorm/RMSNorm.
    """
    
    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: "DeviceMesh") -> None:
        """
        Replicate module parameters to all TP ranks.
        
        For LayerNorm/RMSNorm, parameters need to be consistent across all ranks.
        """
        for p_name, param in module.named_parameters():
            replicated_param = nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(p_name, replicated_param)


# LoRA class mapping
_LORA_CLS_MAP = {}

def _init_lora_cls_map():
    """Lazily initialize LoRA class mapping."""
    global _LORA_CLS_MAP
    if not _LORA_CLS_MAP and _is_torch_tp_available():
        _LORA_CLS_MAP = {
            ColwiseParallel: ColwiseParallelLora,
            RowwiseParallel: RowwiseParallelLora,
            SequenceParallel: SequenceParallelLora,
        }


def translate_to_lora(plan: "ParallelStyle") -> "ParallelStyle":
    """
    Convert standard ParallelStyle to LoRA-aware version.
    
    This function converts by modifying the object's __class__,
    preserving original configuration (like output_layouts, use_local_output, etc.).
    
    Args:
        plan: Original ParallelStyle object
        
    Returns:
        Converted LoRA-aware ParallelStyle object
    """
    if not _is_torch_tp_available():
        return plan
    
    _init_lora_cls_map()
    
    plan_type = type(plan)
    if plan_type in _LORA_CLS_MAP:
        plan.__class__ = _LORA_CLS_MAP[plan_type]
    
    return plan


def is_lora_model(model: nn.Module) -> bool:
    """
    Detect whether a model contains LoRA adapters.
    
    Args:
        model: Model to check
        
    Returns:
        True if the model contains LoRA adapters
    """
    for name, module in model.named_modules():
        # Check for peft LoRA
        if hasattr(module, "lora_A") or hasattr(module, "lora_a"):
            return True
        # Check for peft config
        if hasattr(model, "peft_config"):
            return True
    return False
