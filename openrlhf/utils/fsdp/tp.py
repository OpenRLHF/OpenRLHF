"""
Tensor Parallelism for FSDP2 with Ring Attention Compatibility
================================================================

This module supports both TP-only and TP+SP (SequenceParallel) modes.
When using Ring Attention for Context Parallelism (CP), TP and SP are compatible:

Data Flow with TP SP + Ring Attention CP:
-----------------------------------------
1. LayerNorm (SP):        [B, S/(cp*tp), H] → [B, S/(cp*tp), H]
2. AllGather (PrepareModuleInput): [B, S/(cp*tp), H] → [B, S/cp, H]
3. QKV Projection (TP):   [B, S/cp, H] → [B, S/cp, N/tp, head_dim]
4. Ring Attention (CP):   P2P on CP group (same TP rank) → Works!
5. O Projection (TP):     Reduce-scatter → [B, S/(cp*tp), H]

Key Insight: Ring Attention P2P happens within the same TP rank, so
N/tp is always consistent across CP communication.

Reference: AReaL (areal/utils/fsdp/parallel.py), Automodel
"""

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
# ReplicateParallel Style (from AReaL)
# =============================================================================


class ReplicateParallel(ParallelStyle):
    """Replicate computation style for modules like Q/K norm and score layers.

    This wrapping ensures:
    1. Module parameters are DTensors on the given mesh (for gradient norm clipping)
    2. Proper input/output conversion between Tensor and DTensor

    Reference: AReaL (areal/utils/fsdp/parallel.py)
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


# =============================================================================
# LoRA-Aware Parallel Styles
# =============================================================================


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
    """ColwiseParallel with LoRA support - shards lora_B along dim 0."""

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
            # Called by distribute_module on non-linear submodules (e.g., ModuleDict).
            return

        # Base weight/bias
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
                # Shard LoRA-A like base weight (Automodel style) - requires rank divisible by TP size
                _distribute_param(m, "weight", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                if getattr(m, "bias", None) is not None:
                    _distribute_param(m, "bias", device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                # All-gather LoRA-A output so LoRA-B (also sharded) gets replicated input
                m.register_forward_hook(
                    lambda mod, inp, out, mesh=device_mesh: (
                        out.redistribute(mesh, [Replicate()]) if isinstance(out, DTensor) else out
                    )
                )

    def _partition_embedding_fn(self, name, module, device_mesh):
        # Colwise sharding for embeddings: shard weight on dim 1, preserve requires_grad.
        _distribute_param(module, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)


class RowwiseParallelLora(RowwiseParallel):
    """RowwiseParallel with LoRA support - shards lora_A along dim 1."""

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
            # Called by distribute_module on non-linear submodules (e.g., ModuleDict).
            return

        # Base weight/bias
        _distribute_param(base_layer, "weight", device_mesh, [Shard(1)], src_data_rank=self.src_data_rank)
        if getattr(base_layer, "bias", None) is not None:
            _distribute_param(base_layer, "bias", device_mesh, [Replicate()], src_data_rank=self.src_data_rank)

        lora_a = getattr(module, "lora_A", None) or getattr(module, "lora_a", None)
        lora_b = getattr(module, "lora_B", None) or getattr(module, "lora_b", None)
        # Automodel style: both LoRA-A and LoRA-B use Shard(1), no hook needed
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
        # Rowwise sharding for embeddings: shard weight on dim 0, preserve requires_grad.
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


class SequenceParallelAllGather(SequenceParallelPreserveGrad):
    """SequenceParallel that all-gathers output (for LayerNorm before attention)."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor) and any(isinstance(p, Shard) for p in outputs.placements):
            outputs = outputs.redistribute(device_mesh, [Replicate()])
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


def _to_lora(style):
    """Convert parallel style to LoRA-aware version, preserving layouts."""
    if isinstance(style, ColwiseParallel) and not isinstance(style, ColwiseParallelLora):
        style.__class__ = ColwiseParallelLora
        return style
    if isinstance(style, RowwiseParallel) and not isinstance(style, RowwiseParallelLora):
        style.__class__ = RowwiseParallelLora
        return style
    return style


# =============================================================================
# TP Plans
# =============================================================================

# Attention + MLP layers (shared by all LLaMA-style models)
_ATTN_MLP_PLAN = {
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
}


def _base_plan(sequence_parallel=False, layernorm_cls=SequenceParallel):
    """Base TP plan for LLaMA-style models.

    When sequence_parallel=True:
    - LayerNorm uses SP: input/output are Shard(1) on sequence dim
    - PrepareModuleInput AllGathers before Attention and MLP
    - o_proj and down_proj reduce-scatter back to Shard(1)

    Data flow with SP + Ring Attention CP:
    1. LayerNorm (SP):         [B, S/(cp*tp), H] Shard(1)
    2. PrepareModuleInput:     AllGather → [B, S/cp, H] Replicate
    3. QKV Projection (TP):    → [B, S/cp, N/tp, head_dim]
    4. Ring Attention (CP):    P2P on CP group (same TP rank)
    5. O Projection (TP):      Reduce-scatter → [B, S/(cp*tp), H] Shard(1)

    Reference: AReaL (areal/utils/fsdp/parallel.py)
    """
    plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        **_ATTN_MLP_PLAN,
    }
    if sequence_parallel:
        plan.update(
            {
                # Embedding: output Shard(1) for SP
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                # Final norm: SP
                "model.norm": SequenceParallelPreserveGrad(),
                # LayerNorms: SP (input Shard(1), output Shard(1))
                "model.layers.*.input_layernorm": layernorm_cls(use_local_output=False),
                "model.layers.*.post_attention_layernorm": layernorm_cls(use_local_output=False),
                # PrepareModuleInput: AllGather before Attention (Shard(1) → Replicate)
                # This enables Ring Attention to receive [B, S/cp, H] on each CP rank
                "model.layers.*.self_attn": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                ),
                # O_proj: Reduce-scatter to Shard(1) for residual connection
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                # PrepareModuleInput: AllGather before MLP (Shard(1) → Replicate)
                "model.layers.*.mlp": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                ),
                # down_proj: Reduce-scatter to Shard(1) for residual connection
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                # lm_head: input Shard(1), output Shard(-1) on vocab dim
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
            }
        )
    return plan


def _llama_plan(model, sequence_parallel):
    """LLaMA/Mistral plan with SequenceParallelAllGather for LayerNorm."""
    return _base_plan(sequence_parallel, SequenceParallelAllGather)


def _qwen_plan(model, sequence_parallel):
    """Qwen plan: adds ReplicateParallel for Q/K norm.

    Q/K norms (Qwen3) receive head-sharded input from ColwiseParallel q_proj/k_proj.
    They compute per-token independently, but need ReplicateParallel to:
    1. Ensure parameters are DTensors on the TP mesh (for gradient norm clipping)
    2. Ensure consistent mesh structure across all parameters

    Reference: AReaL (areal/utils/fsdp/parallel.py)
    """
    plan = _base_plan(sequence_parallel, SequenceParallelAllGather)
    # Add Q/K norm handling for Qwen3 - ReplicateParallel ensures DTensor wrapping
    plan.update(
        {
            "model.layers.*.self_attn.q_norm": ReplicateParallel(),
            "model.layers.*.self_attn.k_norm": ReplicateParallel(),
        }
    )
    return plan


# Model class -> plan function
_MODEL_PLANS = {
    "LlamaForCausalLM": _llama_plan,
    "LlamaForSequenceClassification": _llama_plan,
    "MistralForCausalLM": _llama_plan,
    "Qwen2ForCausalLM": _qwen_plan,
    "Qwen3ForCausalLM": _qwen_plan,
}


def get_tp_plan(model, sequence_parallel=False, custom_plan=None):
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


def _str_to_style(s):
    """Convert string to ParallelStyle."""
    styles = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "sequence_parallel": SequenceParallelPreserveGrad(),
        "replicate": ReplicateParallel(),
    }
    return styles[s]


def _add_score_layer_plan(model, plan):
    """Add score layer handling for Critic models (PPO).

    The score layer in Critic models outputs scalar values per token.
    With Ring Attention CP, each rank has [B, S/cp, H] input.
    ReplicateParallel ensures:
    1. Parameters are DTensors on the TP mesh (for gradient norm clipping)
    2. Output is properly formatted

    Reference: AReaL (areal/utils/fsdp/parallel.py)
    """
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        # For PPO's critic model's score layer:
        # - Input is [B, S, H] (or [B, S/cp, H] with Ring Attention)
        # - Score is a linear layer with replicated weights
        # - Output should be replicated across TP ranks
        plan["score"] = ReplicateParallel(
            input_layout=Replicate(),
            desired_input_layout=Replicate(),
            output_layout=Replicate(),
            use_local_output=True,
        )
        logger.info("Added ReplicateParallel for score layer (Critic model)")
    return plan


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
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
        result["lm_head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
    return result


# =============================================================================
# Application
# =============================================================================


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


def apply_tensor_parallel(model, tp_mesh, tp_plan=None, sequence_parallel=False, validate=True, ring_attn_group=None):
    """Apply Tensor Parallelism to model.

    Args:
        model: Model to parallelize
        tp_mesh: DeviceMesh for TP
        tp_plan: Optional custom TP plan
        sequence_parallel: Whether to use TP SequenceParallel for activation memory savings
        validate: Whether to validate head divisibility
        ring_attn_group: If provided, register hooks for DTensor<->Tensor conversion
                         to support Ring Attention (CP) with TP

    Ring Attention + TP SequenceParallel Compatibility:
    ---------------------------------------------------
    Ring Attention and TP SequenceParallel are COMPATIBLE because:
    - TP operates on head dimension (N/tp)
    - CP operates on sequence dimension (S/cp)
    - Ring P2P happens within the same TP rank (mesh structure guarantees this)
    - All CP ranks in a Ring have the same head shard, so [B, S/cp, N/tp, head_dim] is compatible

    Data Flow with SP + Ring Attention:
    1. LayerNorm (SP): [B, S/(cp*tp), H] Shard(1)
    2. PrepareModuleInput: AllGather → [B, S/cp, H] Replicate
    3. AttentionDTensorHook: DTensor → local tensor
    4. Ring Attention: P2P on CP group
    5. AttentionDTensorHook: local tensor → DTensor Replicate
    6. O_proj: Reduce-scatter → [B, S/(cp*tp), H] Shard(1)

    When sequence_parallel=False (simpler mode):
    - Each CP rank computes LayerNorm on [B, S/cp, H] independently
    - No TP AllGather/ReduceScatter for LayerNorm activations
    - Still compatible with Ring Attention
    """
    if tp_mesh.size() == 1:
        return model

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

    # Add score layer handling for Critic models (PPO)
    tp_plan = _add_score_layer_plan(model, tp_plan)

    # Convert to LoRA-aware styles and rely on parallelize_module's fnmatch-based
    # wildcard matching (e.g., 'model.layers.*.self_attn.q_proj').
    tp_plan = {k: _to_lora(v) for k, v in tp_plan.items()}
    parallelize_module(tp_root, tp_mesh, tp_plan)

    # Register hooks for Ring Attention (CP) compatibility
    if ring_attn_group is not None:
        _register_attention_hooks(tp_root, tp_mesh, sequence_parallel=sequence_parallel)

    return model


# =============================================================================
# TP + Ring Attention (CP) Compatibility
# =============================================================================


class AttentionDTensorHook:
    """Hook to convert DTensor to local tensor before attention for Ring Attention compatibility.

    Ring Attention (ring_flash_attn) requires regular Tensors, not DTensor.
    This hook converts DTensor inputs to local tensors before attention,
    and converts the output back to DTensor afterward.

    Execution Order with SequenceParallel:
    --------------------------------------
    1. PrepareModuleInput (if SP enabled): Shard(1) → Replicate (AllGather on TP)
    2. This hook pre_forward: DTensor Replicate → local tensor [B, S/cp, H]
    3. Ring Attention: P2P on CP group, operates on local tensors
    4. This hook post_forward: local tensor → DTensor Replicate
    5. O_proj: DTensor Replicate → DTensor Shard(1) (reduce-scatter)

    Key Insight:
    - Ring Attention P2P happens within the same TP rank (guaranteed by mesh structure)
    - So all CP ranks in a Ring have the same head shard (N/tp)
    - The shapes are always compatible: [B, S/cp, N/tp, head_dim]
    """

    def __init__(self, tp_mesh, sequence_parallel=False):
        self.tp_mesh = tp_mesh
        self.sequence_parallel = sequence_parallel
        self._saved_placements = None

    def pre_forward(self, module, args, kwargs):
        """Convert DTensor inputs to local tensor before attention.

        When SP is enabled, input is already AllGathered to Replicate by PrepareModuleInput.
        We convert to local tensor for Ring Attention to work with regular tensors.
        """
        if not args:
            return args, kwargs

        hidden_states = args[0]
        if isinstance(hidden_states, DTensor):
            # Save placements for post-forward reconstruction
            # With SP: (Replicate,) after PrepareModuleInput AllGather
            # Without SP: (Replicate,) directly
            self._saved_placements = hidden_states.placements
            # Convert to local tensor - Ring Attention needs regular tensors
            local_hidden = hidden_states.to_local()
            return (local_hidden, *args[1:]), kwargs
        return args, kwargs

    def post_forward(self, module, args, output):
        """Convert local tensor output back to DTensor after attention.

        Reconstructs DTensor with the same placements as input (Replicate).
        O_proj will then reduce-scatter to Shard(1) for residual connection.
        """
        if self._saved_placements is None:
            return output

        # Handle tuple output (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        # Reconstruct DTensor from local tensor
        # Placement is Replicate - O_proj will reduce-scatter to Shard(1)
        if not isinstance(attn_output, DTensor):
            attn_output = DTensor.from_local(
                attn_output,
                self.tp_mesh,
                self._saved_placements,
                run_check=False,
            )

        # Clear saved state
        self._saved_placements = None

        if isinstance(output, tuple):
            return (attn_output, *output[1:])
        return attn_output


def _register_attention_hooks(model, tp_mesh, sequence_parallel=False):
    """Register DTensor conversion hooks on attention modules for Ring Attention compatibility.

    This finds all attention modules (matching 'self_attn' in name, excluding projections)
    and registers pre/post forward hooks to handle DTensor<->Tensor conversion.

    Args:
        model: Model to register hooks on
        tp_mesh: TP DeviceMesh
        sequence_parallel: Whether SequenceParallel is enabled

    Note:
        When sequence_parallel=True, PrepareModuleInput will AllGather the input
        before our hook runs. The hook then converts the Replicate DTensor to
        local tensor for Ring Attention to process.
    """
    hook_count = 0
    for name, module in model.named_modules():
        # Match attention modules like 'model.layers.0.self_attn'
        # Exclude sub-modules like 'self_attn.q_proj', 'self_attn.k_norm'
        if "self_attn" in name and not any(
            x in name for x in [".q_proj", ".k_proj", ".v_proj", ".o_proj", ".q_norm", ".k_norm", "_proj", "_norm"]
        ):
            # Check if this is the attention module itself (has forward method and proj attributes)
            if hasattr(module, "q_proj") or hasattr(module, "qkv_proj"):
                hook = AttentionDTensorHook(tp_mesh, sequence_parallel=sequence_parallel)
                module.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
                module.register_forward_hook(hook.post_forward)
                hook_count += 1
                logger.debug(f"Registered DTensor hook on: {name}")

    if hook_count == 0:
        logger.warning("No attention modules found for DTensor hook registration")
    else:
        sp_status = "with SP" if sequence_parallel else "without SP"
        logger.info(f"Registered {hook_count} DTensor conversion hooks for attention modules ({sp_status})")
