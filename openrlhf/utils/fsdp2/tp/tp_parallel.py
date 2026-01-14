"""
Tensor Parallelism for FSDP2
============================

This module provides tensor parallelism support including:
- Custom ParallelStyle variants (LoRA-aware, SequenceParallel)
- TP plans for common HF model families (LLaMA, Qwen, Mistral)
- Model parallelization utilities
- Ring Attention compatibility hooks
"""

from __future__ import annotations

import logging
from functools import partial

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import ParallelStyle, distribute_module
from torch.distributed.tensor.placement_types import Placement

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Parallel Styles
# =============================================================================


class ReplicateParallel(ParallelStyle):
    """Replicate computation style for modules that should not be sharded.

    `distribute_module(..., partition_fn=None)` will replicate parameters/buffers to DTensor,
    which keeps mesh metadata consistent across parameters (useful for global grad norm).
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        desired_input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = desired_input_layout or Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (input_layout,), run_check=False)

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(placements=(desired_input_layout,), async_op=True)
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(self._prepare_input_fn, self.input_layout, self.desired_input_layout),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


def _iter_modules(x):
    if x is None:
        return []
    if isinstance(x, nn.ModuleDict):
        return list(x.values())
    return [x]


def _get_base_linear(module: nn.Module) -> nn.Linear | None:
    base = getattr(module, "get_base_layer", None)
    if callable(base):
        try:
            base_layer = module.get_base_layer()
        except Exception:
            base_layer = None
        if isinstance(base_layer, nn.Linear):
            return base_layer
    return module if isinstance(module, nn.Linear) else None


def _distribute_param(module, name, mesh, placements, *, src_data_rank: int = 0):
    param = getattr(module, name, None)
    if param is None or isinstance(param, DTensor):
        return
    setattr(
        module,
        name,
        nn.Parameter(
            distribute_tensor(param, mesh, placements, src_data_rank=src_data_rank),
            requires_grad=param.requires_grad,
        ),
    )


class ColwiseParallelLora(ColwiseParallel):
    """ColwiseParallel that also shards PEFT LoRA linear wrappers."""

    src_data_rank: int = 0

    def _apply(self, module: nn.Module, device_mesh):
        if isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        elif _get_base_linear(module) is not None:
            partition_fn = self._partition_linear_fn
        else:
            raise NotImplementedError(
                f"{type(self).__name__} only supports nn.Linear/nn.Embedding (and PEFT LoRA Linear wrappers), got {type(module)}"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )

    def _partition_linear_fn(self, name, module, device_mesh):
        base_layer = _get_base_linear(module)
        if base_layer is None:
            return

        _distribute_param(base_layer, "weight", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        if getattr(base_layer, "bias", None) is not None:
            _distribute_param(base_layer, "bias", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)

        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)
        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        if lora_b:
            for m in _iter_modules(lora_b):
                _distribute_param(m, "weight", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                if getattr(m, "bias", None) is not None:
                    _distribute_param(m, "bias", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        if lora_a:
            for m in _iter_modules(lora_a):
                _distribute_param(m, "weight", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                if getattr(m, "bias", None) is not None:
                    _distribute_param(m, "bias", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                # All-gather LoRA-A output so LoRA-B (also sharded) gets replicated input.
                m.register_forward_hook(
                    lambda mod, inp, out, mesh=device_mesh: (
                        out.redistribute(mesh, [Replicate()]) if isinstance(out, DTensor) else out
                    )
                )

    def _partition_embedding_fn(self, name, module, device_mesh):
        _distribute_param(module, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)


class RowwiseParallelLora(RowwiseParallel):
    """RowwiseParallel that also shards PEFT LoRA linear wrappers."""

    src_data_rank: int = 0

    def _apply(self, module: nn.Module, device_mesh):
        if isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
            self.desired_input_layouts = (Replicate(),)
        elif _get_base_linear(module) is not None:
            partition_fn = self._partition_linear_fn
            self.desired_input_layouts = (Shard(-1),)
        else:
            raise NotImplementedError(
                f"{type(self).__name__} only supports nn.Linear/nn.Embedding (and PEFT LoRA Linear wrappers), got {type(module)}"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )

    def _partition_linear_fn(self, name, module, device_mesh):
        base_layer = _get_base_linear(module)
        if base_layer is None:
            return

        _distribute_param(base_layer, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)
        if getattr(base_layer, "bias", None) is not None:
            _distribute_param(base_layer, "bias", device_mesh, [Replicate()], src_data_rank=self.src_data_rank)

        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)
        if lora_a:
            for m in _iter_modules(lora_a):
                _distribute_param(m, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)
                if getattr(m, "bias", None) is not None:
                    _distribute_param(m, "bias", device_mesh, [Replicate()], src_data_rank=self.src_data_rank)
        if lora_b:
            for m in _iter_modules(lora_b):
                _distribute_param(m, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)
                if getattr(m, "bias", None) is not None:
                    _distribute_param(m, "bias", device_mesh, [Replicate()], src_data_rank=self.src_data_rank)

    def _partition_embedding_fn(self, name, module, device_mesh):
        _distribute_param(module, "weight", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)


class SequenceParallelPreserveGrad(SequenceParallel):
    """SequenceParallel that preserves requires_grad when re-wrapping parameters."""

    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh):
        for p_name, param in module.named_parameters():
            module.register_parameter(
                p_name,
                nn.Parameter(
                    DTensor.from_local(param, device_mesh, [Replicate()], run_check=False),
                    requires_grad=param.requires_grad,
                ),
            )


def ensure_lora_style(style: ParallelStyle) -> ParallelStyle:
    """Upgrade base torch TP styles to LoRA-aware variants (without mutating in-place)."""

    if isinstance(style, ColwiseParallel) and not isinstance(style, ColwiseParallelLora):
        return ColwiseParallelLora(
            input_layouts=style.input_layouts[0],
            output_layouts=style.output_layouts[0],
            use_local_output=style.use_local_output,
        )
    if isinstance(style, RowwiseParallel) and not isinstance(style, RowwiseParallelLora):
        return RowwiseParallelLora(
            input_layouts=style.input_layouts[0],
            output_layouts=style.output_layouts[0],
            use_local_output=style.use_local_output,
        )
    return style


# =============================================================================
# TP Plans for HF Models
# =============================================================================


def _attn_mlp_plan():
    """Attention + MLP layers (shared by LLaMA-style models)."""
    return {
        "model.layers.*.self_attn.q_proj": ColwiseParallelLora(),
        "model.layers.*.self_attn.k_proj": ColwiseParallelLora(),
        "model.layers.*.self_attn.v_proj": ColwiseParallelLora(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallelLora(),
        "model.layers.*.self_attn.o_proj": RowwiseParallelLora(),
        "model.layers.*.mlp.gate_proj": ColwiseParallelLora(),
        "model.layers.*.mlp.up_proj": ColwiseParallelLora(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallelLora(),
        "model.layers.*.mlp.down_proj": RowwiseParallelLora(),
    }


def _base_plan(sequence_parallel: bool = False, layernorm_cls=SequenceParallel):
    """Base TP plan for LLaMA-style models."""

    plan = {
        "model.embed_tokens": RowwiseParallelLora(input_layouts=Replicate()),
        # Return full logits (all-gather vocab) as a local tensor for simple loss functions.
        "lm_head": ColwiseParallelLora(output_layouts=Replicate(), use_local_output=True),
        **_attn_mlp_plan(),
    }

    if sequence_parallel:
        plan.update(
            {
                # Embedding: output Shard(1) for SP
                "model.embed_tokens": RowwiseParallelLora(input_layouts=Replicate(), output_layouts=Shard(1)),
                # Final norm: SP (preserve requires_grad)
                "model.norm": SequenceParallelPreserveGrad(),
                # LayerNorms: SP (input Shard(1), output Shard(1) as local tensor)
                "model.layers.*.input_layernorm": layernorm_cls(use_local_output=True),
                "model.layers.*.post_attention_layernorm": layernorm_cls(use_local_output=True),
                # AllGather before Attention / MLP: Shard(1) -> Replicate
                "model.layers.*.self_attn": PrepareModuleInput(
                    input_kwarg_layouts={"hidden_states": Shard(1)},
                    desired_input_kwarg_layouts={"hidden_states": Replicate()},
                ),
                "model.layers.*.mlp": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                ),
                # Reduce-scatter back to Shard(1) for residual connections
                "model.layers.*.self_attn.o_proj": RowwiseParallelLora(output_layouts=Shard(1)),
                "model.layers.*.mlp.down_proj": RowwiseParallelLora(output_layouts=Shard(1)),
                # lm_head: input Shard(1), output Replicate() (full vocab logits)
                "lm_head": ColwiseParallelLora(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                    use_local_output=True,
                ),
            }
        )

    return plan


def _llama_plan(model, sequence_parallel: bool):
    return _base_plan(sequence_parallel, SequenceParallelPreserveGrad)


def _qwen_plan(model, sequence_parallel: bool):
    plan = _base_plan(sequence_parallel, SequenceParallelPreserveGrad)
    plan.update(
        {
            # Q/K norms receive head-sharded input; keep them replicated.
            "model.layers.*.self_attn.q_norm": ReplicateParallel(),
            "model.layers.*.self_attn.k_norm": ReplicateParallel(),
        }
    )
    return plan


_MODEL_PLANS = {
    "LlamaForCausalLM": _llama_plan,
    "LlamaForSequenceClassification": _llama_plan,
    "MistralForCausalLM": _llama_plan,
    "Qwen2ForCausalLM": _qwen_plan,
    "Qwen3ForCausalLM": _qwen_plan,
}


def _str_to_style(s: str):
    styles = {
        "colwise": ColwiseParallelLora(),
        "rowwise": RowwiseParallelLora(),
        "colwise_rep": ColwiseParallelLora(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallelLora(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallelPreserveGrad(),
        "replicate": ReplicateParallel(),
    }
    return styles[s]


def _get_hf_plan(model):
    """Extract TP plan from HuggingFace model's _tp_plan attribute."""
    hf = {}
    for src in [type(model), model, getattr(model, "model", None)]:
        if src and hasattr(src, "_tp_plan"):
            prefix = "model." if src == getattr(model, "model", None) else ""
            hf.update({f"{prefix}{k}": v for k, v in src._tp_plan.items()})
    if not hf:
        return None

    hf.setdefault("model.embed_tokens", "rowwise_rep")
    result = {k: _str_to_style(v) if isinstance(v, str) else v for k, v in hf.items()}
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallelLora):
        result["lm_head"] = ColwiseParallelLora(output_layouts=Shard(-1), use_local_output=False)
    return result


def get_tp_plan(model, sequence_parallel: bool = False, custom_plan=None):
    """Get TP plan: custom > model-specific > HuggingFace > base."""
    if custom_plan:
        return {k: _str_to_style(v) if isinstance(v, str) else v for k, v in custom_plan.items()}

    cls_name = type(model).__name__
    if cls_name in _MODEL_PLANS:
        try:
            return _MODEL_PLANS[cls_name](model, sequence_parallel)
        except Exception as e:
            logger.warning(f"Plan failed for {cls_name}: {e}")

    hf_plan = _get_hf_plan(model)
    if hf_plan:
        return hf_plan

    return _base_plan(sequence_parallel)


def maybe_add_score_layer_plan(model, plan: dict):
    """Add score layer handling for Critic models (PPO)."""
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        plan = dict(plan)
        plan["score"] = ReplicateParallel(
            input_layout=Replicate(),
            desired_input_layout=Replicate(),
            output_layout=Replicate(),
            use_local_output=True,
        )
        logger.info("Added ReplicateParallel for score layer (Critic model)")
    return plan


# =============================================================================
# Apply TP to Model
# =============================================================================


def maybe_enable_async_tp(tp_mesh: DeviceMesh, enabled: bool = False):
    """Enable Async Tensor Parallel (Symmetric Memory) for NVLink environments.

    See torchtitan for reference. This requires:
    1. torch.compile support
    2. NVLink / Symmetric Memory support
    """
    if not enabled:
        return

    try:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        import torch._inductor.config

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)
        logger.info("Enabled Async TP (Symmetric Memory)")
    except Exception as e:
        logger.warning(f"Failed to enable Async TP: {e}")


def _tie_weights(model: nn.Module):
    """Ensure tied weights remain tied after TP sharding.

    In many HF models, lm_head.weight is tied to embed_tokens.weight.
    Parallelizing them separately may create independent DTensor objects.
    """
    if getattr(model.config, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
        logger.info("Re-tied lm_head and embed_tokens weights after TP")


def validate_tp_mesh(model, tp_mesh):
    """Validate attention heads divisible by TP size."""
    if tp_mesh.size() == 1:
        return
    cfg = getattr(model, "config", None)
    if not cfg:
        return
    tp = tp_mesh.size()
    heads = getattr(cfg, "num_attention_heads", 0)
    kv = getattr(cfg, "num_key_value_heads", 0) or heads
    if heads and heads % tp:
        raise ValueError(f"num_attention_heads ({heads}) not divisible by TP ({tp})")
    if kv and kv % tp:
        raise ValueError(f"num_key_value_heads ({kv}) not divisible by TP ({tp})")


def apply_tensor_parallel(
    model,
    tp_mesh,
    tp_plan=None,
    sequence_parallel: bool = False,
    validate: bool = True,
    ring_attn_group=None,
    enable_async_tp: bool = False,
):
    """Apply Tensor Parallelism (DTensor parallel) to a model."""

    if tp_mesh.size() == 1:
        return model

    maybe_enable_async_tp(tp_mesh, enabled=enable_async_tp)

    # If using PEFT, apply TP on the underlying HF model (keys like `model.layers.*`),
    # but keep returning the original wrapper.
    tp_root = model
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            tp_root = model.get_base_model()
    except Exception:
        tp_root = model

    if validate:
        validate_tp_mesh(tp_root, tp_mesh)
    if tp_plan is None:
        tp_plan = get_tp_plan(tp_root, sequence_parallel)

    tp_plan = maybe_add_score_layer_plan(model, tp_plan)

    # Ensure plan is LoRA-aware even if user/HF provides base styles.
    tp_plan = {k: ensure_lora_style(v) for k, v in tp_plan.items()}

    parallelize_module(tp_root, tp_mesh, tp_plan)

    # Re-tie weights if necessary (must happen after parallelize_module)
    _tie_weights(tp_root)

    if ring_attn_group is not None:
        register_attention_hooks(tp_root, tp_mesh, sequence_parallel=sequence_parallel)

    return model


# =============================================================================
# Ring Attention Compatibility
# =============================================================================


class AttentionDTensorHook:
    """Convert DTensor activations to local tensors for Ring Attention compatibility.

    Ring Attention kernels expect regular local tensors; with TP(+SP) wrappers, attention
    may receive DTensor inputs. We convert inputs to local tensors and ensure outputs
    are local as well (do not reconstruct DTensor outputs here).
    """

    def __init__(self, tp_mesh: DeviceMesh, sequence_parallel: bool = False):
        self.tp_mesh = tp_mesh
        self.sequence_parallel = sequence_parallel

    def pre_forward(self, module, args, kwargs):
        if args:
            hidden_states = args[0]
            if isinstance(hidden_states, DTensor):
                return (hidden_states.to_local(), *args[1:]), kwargs
            return args, kwargs

        hidden_states = kwargs.get("hidden_states")
        if isinstance(hidden_states, DTensor):
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden_states.to_local()
            return args, new_kwargs
        return args, kwargs

    def post_forward(self, module, args, output):
        if isinstance(output, DTensor):
            return output.to_local()
        if isinstance(output, tuple) and output and isinstance(output[0], DTensor):
            return (output[0].to_local(), *output[1:])
        return output


def register_attention_hooks(model: nn.Module, tp_mesh: DeviceMesh, sequence_parallel: bool = False) -> None:
    """Register DTensor conversion hooks on attention modules for Ring Attention compatibility."""

    hook_count = 0
    for name, module in model.named_modules():
        if "self_attn" in name and not any(
            x in name for x in [".q_proj", ".k_proj", ".v_proj", ".o_proj", ".q_norm", ".k_norm", "_proj", "_norm"]
        ):
            if hasattr(module, "q_proj") or hasattr(module, "qkv_proj"):
                if getattr(module, "_openrlhf_tp_cp_hook_registered", False):
                    continue
                hook = AttentionDTensorHook(tp_mesh, sequence_parallel=sequence_parallel)
                module.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
                module.register_forward_hook(hook.post_forward)
                setattr(module, "_openrlhf_tp_cp_hook_registered", True)
                hook_count += 1
                logger.debug(f"Registered DTensor hook on: {name}")

    if hook_count == 0:
        logger.warning("No attention modules found for DTensor hook registration")
    else:
        sp_status = "with SP" if sequence_parallel else "without SP"
        logger.info(f"Registered {hook_count} DTensor conversion hooks for attention modules ({sp_status})")
