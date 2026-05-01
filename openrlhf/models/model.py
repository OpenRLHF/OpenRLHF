import copy
import gc
import inspect
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

from openrlhf.utils.fsdp.packing import (
    cp_dtensor_full_sequence,
    is_automodel_custom_model,
    pack_padded_batch,
    pad_to_cp_multiple,
    unpack_to_padded,
    unshard_dtensor,
)
from openrlhf.utils.logging_utils import init_logger

from .actor import (
    _build_peft_config_dict,
    _detect_moe_arch,
    _has_hf_flash_attn_2,
    _move_model_to_cpu_for_offload,
    _patch_nemo_automodel_runtime,
    _resolve_custom_backend_attn,
    _validate_attn_implementation,
    _will_use_hf_model,
)

logger = init_logger(__name__)


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    param_dtype: str = "bf16",
    lora_rank: int = 0,
    lora_alpha: int = 16,
    target_modules=None,
    lora_dropout: float = 0,
    normalize_reward: bool = False,
    attn_implementation: str = "flash_attention_2",
    device_mesh=None,
    moe_mesh=None,
    distributed_config=None,
    moe_config=None,
    activation_checkpointing: bool = False,
    init_value_head: bool = False,
    value_head_prefix: str = "score",
    packing_samples: bool = False,
    force_hf_model: bool = False,
    use_liger_kernel: bool = False,
    use_fp32_master_weights: Optional[bool] = None,
    **kwargs,
) -> nn.Module:
    """Build a reward or critic model with an AutoModel-managed regression head."""
    assert model_type in ("critic", "reward"), f"invalid model_type: {model_type}"

    te_unavailable_reason = _validate_sequence_regression_attn(attn_implementation)
    mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
    cp_size = device_mesh["cp"].size() if device_mesh is not None and "cp" in mesh_dims else 1
    if packing_samples:
        if cp_size > 1:
            raise NotImplementedError(
                "--fsdp.packing_samples for reward/critic sequence-regression currently supports CP size 1 only. "
                "AutoModel custom TE packed value heads need an upstream custom "
                "SequenceClassification/value-head path."
            )
        if attn_implementation not in {"te", "flash_attention_2"}:
            raise NotImplementedError(
                "--fsdp.packing_samples for reward/critic sequence-regression currently requires "
                "--fsdp.attn_implementation te or flash_attention_2. AutoModel custom TE packing is available "
                "for CausalLM actor/reference forwards; incompatible reward/critic models fall back to "
                "HF flash_attention_2 packing."
            )
        if not _has_hf_flash_attn_2():
            raise ValueError(
                "--fsdp.packing_samples with reward/critic sequence-regression falls back to HF "
                "flash_attention_2 packing and requires flash-attn to be installed."
            )

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.num_labels = 1
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    config.value_head_prefix = value_head_prefix
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")
    custom_sequence_regression_arch = _register_openrlhf_llama_sequence_regression(config)

    from openrlhf.utils.utils import convert_to_torch_dtype, ensure_torchvision_nms_stub

    compute_dtype = convert_to_torch_dtype(param_dtype)
    if use_fp32_master_weights is None:
        use_fp32_master_weights = model_type != "reward"
    torch_dtype = compute_dtype if not use_fp32_master_weights else torch.float32
    # HF MoE sequence-classification checkpoints can mix bf16 experts with fp32
    # router/gate params. FSDP requires uniform original param dtype, so mirror
    # the actor path and avoid fp32 master weights for this case.
    if torch_dtype == torch.float32 and _detect_moe_arch(model_name_or_path):
        torch_dtype = compute_dtype
    peft_config = None
    if lora_rank > 0:
        peft_config = _build_peft_config_dict(lora_rank, lora_alpha, lora_dropout, target_modules)

    ensure_torchvision_nms_stub()
    if use_liger_kernel:
        mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
        tp_size = device_mesh["tp"].size() if "tp" in mesh_dims else 1
        cp_size = device_mesh["cp"].size() if "cp" in mesh_dims else 1
        if tp_size > 1 or cp_size > 1:
            print(
                f"[Liger] AutoModel disables Liger Kernel when TP>1 ({tp_size}) "
                f"or CP>1 ({cp_size}); --fsdp.use_liger_kernel will be a no-op."
            )
    from nemo_automodel import NeMoAutoModelForSequenceClassification

    model_extra_kwargs = dict(kwargs)
    for reserved_kwarg in ("attn_implementation", "config", "force_hf", "has_packed_sequence"):
        model_extra_kwargs.pop(reserved_kwarg, None)

    common_model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_mesh": device_mesh,
        "moe_mesh": moe_mesh,
        "distributed_config": distributed_config,
        "moe_config": moe_config,
        "activation_checkpointing": activation_checkpointing,
        "peft_config": peft_config,
        "use_liger_kernel": use_liger_kernel,
        **model_extra_kwargs,
    }

    def load_model(*, force_hf: bool, attn_impl: str, has_packed_sequence: bool, extra_kwargs=None) -> nn.Module:
        attempt_config = copy.deepcopy(config)
        attempt_config._attn_implementation = attn_impl
        if not force_hf and custom_sequence_regression_arch is not None:
            attempt_config.architectures = [custom_sequence_regression_arch]
        return NeMoAutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=attempt_config,
            attn_implementation=attn_impl,
            has_packed_sequence=has_packed_sequence,
            force_hf=force_hf,
            **common_model_kwargs,
            **(extra_kwargs or {}),
        )

    custom_skip_reason = None
    fallback_reason = None
    model = None
    selected_attn_implementation = None

    use_hf_model = True if force_hf_model else _will_use_hf_model(model_name_or_path, False)
    try_custom = not force_hf_model and not use_hf_model
    if use_hf_model and not force_hf_model:
        architectures = getattr(config, "architectures", None) or []
        arch_desc = ", ".join(architectures) if architectures else type(config).__name__
        custom_skip_reason = f"AutoModel selected the HF sequence-regression implementation for {arch_desc}"
    if packing_samples and try_custom:
        try_custom = False
        custom_skip_reason = (
            "AutoModel custom packed sequence-regression is not wired safely yet; "
            "reward/critic packed batches use HF FlashAttention2 varlen boundaries"
        )
    elif te_unavailable_reason is not None and try_custom:
        try_custom = False
        custom_skip_reason = te_unavailable_reason

    if try_custom:
        try:
            _patch_nemo_automodel_runtime()
            custom_extra_kwargs, custom_attn_implementation = _custom_sequence_regression_backend_kwargs(
                attn_implementation
            )
            model = load_model(
                force_hf=False,
                attn_impl=custom_attn_implementation,
                has_packed_sequence=False,
                extra_kwargs=custom_extra_kwargs,
            )
            incompatibility = _custom_sequence_regression_incompatibility(
                model, value_head_prefix=value_head_prefix, packing_samples=False
            )
            if incompatibility is not None:
                raise _SequenceRegressionFallback(incompatibility)
            selected_attn_implementation = custom_attn_implementation
            if is_automodel_custom_model(model):
                print("[SequenceRegression] Using AutoModel custom reward/critic path.")
        except _SequenceRegressionFallback as exc:
            fallback_reason = str(exc)
            del model
            model = None
            _release_cuda_cache()
        except (AttributeError, ImportError, NotImplementedError, TypeError, ValueError) as exc:
            fallback_reason = f"AutoModel custom reward/critic path failed: {exc}"
            del model
            model = None
            _release_cuda_cache()

    if model is None:
        hf_attn_implementation = _resolve_hf_sequence_regression_attn(attn_implementation, packing_samples)
        if hf_attn_implementation == "flash_attention_2" and not _has_hf_flash_attn_2():
            print("[Attn] flash_attn not installed; downgrading flash_attention_2 to sdpa.")
            hf_attn_implementation = "sdpa"
        if custom_skip_reason is not None:
            print(f"[SequenceRegression] {custom_skip_reason}; using HF fallback.")
        elif fallback_reason is not None:
            print(f"[SequenceRegression] {fallback_reason}; using HF fallback.")
        model = load_model(force_hf=True, attn_impl=hf_attn_implementation, has_packed_sequence=packing_samples)
        selected_attn_implementation = hf_attn_implementation
        if packing_samples:
            print("[Packing] Using HF FlashAttention2 varlen packed path for reward/critic.")

    # FSDP2 can return child module outputs in fp32 even when parameters are
    # bf16. Mirror NeMo-RL by running sequence-regression forwards under the
    # compute dtype so the score head receives matching activations.
    forward_autocast_dtype = compute_dtype if compute_dtype != torch.float32 else None

    if "output_router_logits" in model.config.to_dict():
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
    model.config.use_cache = False
    model.config.normalize_reward = normalize_reward
    model.config.value_head_prefix = value_head_prefix

    if init_value_head:
        _init_regression_head(model, value_head_prefix)

    model = _move_model_to_cpu_for_offload(model, distributed_config)

    wrapper_cls = RewardModel if model_type == "reward" else CriticModel
    wrapper = wrapper_cls(
        model,
        value_head_prefix,
        normalize_reward,
        forward_autocast_dtype,
        packing_samples,
        device_mesh=device_mesh,
    )
    # Mirror iter-32 fix on Actor: stash peft_config on the wrapper so
    # strategy.save_model can forward it to AutoModel's PEFT save addon, which
    # needs it to write adapter_config.json (otherwise AttributeError on dim).
    wrapper.peft_config = peft_config
    return wrapper


class _SequenceRegressionFallback(Exception):
    """Internal signal that the custom AutoModel path is not usable here."""


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _register_openrlhf_llama_sequence_regression(config) -> Optional[str]:
    """Register a causal Llama token-regression AutoModel class for RM/value heads."""
    architectures = getattr(config, "architectures", None) or []
    if getattr(config, "model_type", None) != "llama":
        return None
    if architectures and architectures[0] not in {
        "RewardModel",
        "LlamaForCausalLM",
        "LlamaForSequenceClassification",
        "OpenRLHFLlamaForCausalSequenceRegression",
    }:
        return None

    try:
        from nemo_automodel._transformers.registry import ModelRegistry
        from nemo_automodel.components.distributed.optimized_tp_plans import PARALLELIZE_FUNCTIONS, _parallelize_llama
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
        from nemo_automodel.components.models.llama.model import LlamaModel, LlamaPreTrainedModel
        from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
        from transformers import LlamaConfig
        from transformers.cache_utils import Cache
        from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
        from transformers.processing_utils import Unpack
        from transformers.utils import TransformersKwargs
    except Exception as exc:
        logger.warning("Unable to register OpenRLHF Llama sequence-regression AutoModel path: %s", exc)
        return None

    class OpenRLHFLlamaForCausalSequenceRegression(HFCheckpointingMixin, LlamaPreTrainedModel):
        config_class = LlamaConfig
        base_model_prefix = "model"
        _openrlhf_automodel_custom = True

        @classmethod
        def from_config(cls, config: LlamaConfig, backend: Optional[BackendConfig] = None, **kwargs):
            return cls(config, backend, **kwargs)

        def __init__(self, config: LlamaConfig, backend: Optional[BackendConfig] = None):
            super().__init__(config)
            self.config = config
            self.backend = backend or BackendConfig()
            self.num_labels = getattr(config, "num_labels", 1)
            self.model = LlamaModel(config=config, backend=self.backend)
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
            self.state_dict_adapter = LlamaStateDictAdapter(config=self.config)
            self.post_init()

            if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                self.to(dtype=config.torch_dtype)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(
                    "[OpenRLHFLlamaForCausalSequenceRegression] "
                    f"Attention implementation: {self.config._attn_implementation}"
                )
                print("[OpenRLHFLlamaForCausalSequenceRegression] Custom token regression implementation")
                print(f"[OpenRLHFLlamaForCausalSequenceRegression] torch_dtype: {self.config.torch_dtype}")

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def set_decoder(self, decoder):
            self.model = decoder

        def get_decoder(self):
            return self.model

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
        ) -> SequenceClassifierOutputWithPast:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs: BaseModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            # The regression head is replicated under TP. Score full hidden
            # states so every TP rank computes the same head gradients.
            last_hidden_state = unshard_dtensor(last_hidden_state)
            score_weight = getattr(self.score, "weight", None)
            score_dtype = getattr(score_weight, "dtype", None)
            if score_dtype is not None and last_hidden_state.dtype != score_dtype:
                last_hidden_state = last_hidden_state.to(score_dtype)
            scores = self.score(last_hidden_state)
            out = SequenceClassifierOutputWithPast(
                loss=None,
                logits=scores,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            if return_dict:
                return out
            return out.to_tuple()

    def _parallelize_openrlhf_llama_sequence_regression(model, sequence_parallel: bool = False):
        plan = _parallelize_llama(model, sequence_parallel)
        plan.pop("lm_head", None)
        return plan

    arch_name = "OpenRLHFLlamaForCausalSequenceRegression"
    registry = ModelRegistry.model_arch_name_to_cls
    try:
        existing = registry[arch_name] if arch_name in registry else None
    except Exception:
        existing = None
    if existing is None:
        registry[arch_name] = OpenRLHFLlamaForCausalSequenceRegression
        PARALLELIZE_FUNCTIONS[OpenRLHFLlamaForCausalSequenceRegression] = (
            _parallelize_openrlhf_llama_sequence_regression
        )
    elif getattr(existing, "_openrlhf_automodel_custom", False):
        OpenRLHFLlamaForCausalSequenceRegression = existing

    if "RewardModel" not in registry:
        registry["RewardModel"] = OpenRLHFLlamaForCausalSequenceRegression
    return arch_name


def _validate_sequence_regression_attn(attn_implementation: str) -> Optional[str]:
    try:
        _validate_attn_implementation(attn_implementation)
    except ValueError as exc:
        # Reward/critic can still fall back to HF sdpa/flash_attention_2 when
        # the requested custom-only TE backend is unavailable.
        if attn_implementation == "te" and "transformer-engine" in str(exc):
            return str(exc)
        raise
    return None


def _custom_sequence_regression_backend_kwargs(attn_implementation: str):
    backend_attn = _resolve_custom_backend_attn(attn_implementation, packing_samples=False)
    from nemo_automodel.components.models.common.utils import BackendConfig

    using_te = backend_attn == "te"
    backend_kwargs = {"attn": backend_attn, "rope_fusion": using_te}
    if not using_te:
        backend_kwargs["linear"] = "torch"
        backend_kwargs["experts"] = "torch_mm"
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Attn] AutoModel sequence-regression backend={backend_attn}; config attn_implementation=sdpa.")
    return {"backend": BackendConfig(**backend_kwargs)}, "sdpa"


def _resolve_hf_sequence_regression_attn(attn_implementation: str, packing_samples: bool) -> str:
    if packing_samples:
        return "flash_attention_2"
    if attn_implementation in {"flex", "te"}:
        print(f"[Attn] HF reward/critic fallback does not use {attn_implementation}; using sdpa.")
        return "sdpa"
    return attn_implementation


def _custom_sequence_regression_incompatibility(
    model: nn.Module, *, value_head_prefix: str, packing_samples: bool
) -> Optional[str]:
    if not is_automodel_custom_model(model):
        return None
    if packing_samples:
        return "AutoModel custom reward/critic packing is not supported"
    try:
        _get_regression_head(model, value_head_prefix)
    except AttributeError as exc:
        return str(exc)
    for kwarg in ("output_hidden_states", "return_dict"):
        if not _forward_accepts_kwarg(model, kwarg):
            return f"AutoModel custom reward/critic forward does not accept `{kwarg}`"
    return None


def _forward_accepts_kwarg(model: nn.Module, kwarg: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return True
    if kwarg in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _init_regression_head(model: nn.Module, value_head_prefix: str) -> None:
    head = _get_regression_head(model, value_head_prefix)
    if not hasattr(head, "weight"):
        raise AttributeError(f"`{value_head_prefix}` does not expose a weight parameter")
    with torch.no_grad():
        head.weight.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))


def _get_regression_head(model: nn.Module, value_head_prefix: str) -> nn.Module:
    if hasattr(model, value_head_prefix):
        return getattr(model, value_head_prefix)
    if value_head_prefix != "score" and hasattr(model, "score"):
        return getattr(model, "score")
    raise AttributeError(
        f"Sequence-classification model {type(model).__name__} has no `{value_head_prefix}` head. "
        "Use --fsdp.value_head_prefix to match the checkpoint head name."
    )


class _SequenceRegressionBase(nn.Module):
    """OpenRLHF reward/critic forward semantics over an AutoModel model.

    The trainable head lives inside ``self.model``. This wrapper owns only
    non-persistent runtime buffers, so FSDP, checkpointing, and optimizer state
    should operate on ``self.model``.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head_prefix: str,
        normalize_reward: bool,
        forward_autocast_dtype: Optional[torch.dtype] = None,
        packing_samples: bool = False,
        device_mesh=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.value_head_prefix = value_head_prefix
        self.normalize_reward = normalize_reward
        self._forward_autocast_dtype = forward_autocast_dtype
        self.packing_samples = packing_samples
        self.device_mesh = device_mesh
        mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
        self.cp_mesh = device_mesh["cp"] if device_mesh is not None and "cp" in mesh_dims else None
        self.cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1

        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)
        if hasattr(self.config, "mean"):
            self.mean[0] = self.config.mean
            self.std[0] = self.config.std

    def get_base_model_for_fsdp(self) -> nn.Module:
        return self.model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _token_values(self, input_ids, attention_mask, cp_local_values: bool = False, cp_context_stack=None):
        batch, seqlen = input_ids.shape
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_input_ids = input_ids
        model_attention_mask = attention_mask
        indices = None
        attn_kwargs = {}
        cp_forward = False
        cp_ctx_factory = nullcontext
        if self.packing_samples:
            model_input_ids, position_ids, _, indices, attn_kwargs = pack_padded_batch(
                input_ids, attention_mask, style="hf"
            )
            model_attention_mask = None
        elif self.cp_size > 1:
            from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx

            model_input_ids = pad_to_cp_multiple(input_ids, self.cp_size, seq_dim=1, value=0)
            model_attention_mask = pad_to_cp_multiple(attention_mask, self.cp_size, seq_dim=1, value=0)
            position_ids = pad_to_cp_multiple(position_ids, self.cp_size, seq_dim=1, value=1)
            cp_batch = {
                "input_ids": model_input_ids,
                "attention_mask": model_attention_mask,
                "position_ids": position_ids,
                "labels": model_input_ids,
            }
            cp_ctx_factory, cp_batch = make_cp_batch_and_ctx(self.device_mesh, cp_batch)
            model_input_ids = cp_batch["input_ids"]
            position_ids = cp_batch.get("position_ids")
            model_attention_mask = cp_batch.get("attention_mask")
            cp_forward = True

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._forward_autocast_dtype)
            if self._forward_autocast_dtype is not None and input_ids.is_cuda
            else nullcontext()
        )

        entered_cp_context = False

        def _forward_context():
            nonlocal entered_cp_context
            forward_ctx = cp_ctx_factory()
            if cp_context_stack is not None and cp_forward:
                if not entered_cp_context:
                    # AutoModel CP train context installs backward hooks, so
                    # training code keeps it alive until loss.backward() ends.
                    cp_context_stack.enter_context(forward_ctx)
                    entered_cp_context = True
                return nullcontext()
            return forward_ctx

        def _model_forward(output_hidden_states: bool):
            with _forward_context():
                with autocast_ctx:
                    return self.model(
                        input_ids=model_input_ids,
                        attention_mask=model_attention_mask,
                        position_ids=position_ids,
                        output_hidden_states=output_hidden_states,
                        use_cache=False,
                        return_dict=True,
                        **attn_kwargs,
                    )

        outputs = _model_forward(output_hidden_states=False)
        logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
        if logits is not None and logits.ndim >= 3 and logits.shape[-1] == 1:
            values = unshard_dtensor(logits).squeeze(-1)
        else:
            # Most reward/critic paths are real sequence-regression models and
            # return token logits directly. Only fall back to hidden states for
            # legacy wrappers that expose the value head outside the base model.
            outputs = _model_forward(output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Sequence regression requires logits or hidden_states; model returned neither.")
            last_hidden_states = unshard_dtensor(hidden_states[-1])
            values = _get_regression_head(self.model, self.value_head_prefix)(last_hidden_states).squeeze(-1)
        if self.packing_samples:
            values = unpack_to_padded(values, indices, batch, seqlen)
        elif cp_forward and not cp_local_values and values.shape[1] != seqlen:
            values = cp_dtensor_full_sequence(values, self.cp_mesh, seq_dim=1)
        if cp_forward and not cp_local_values:
            values = values[:, :seqlen]
        return values.float(), outputs


class RewardModel(_SequenceRegressionBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        pad_sequence: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        values, outputs = self._token_values(input_ids, attention_mask)
        reward = values.gather(dim=1, index=eos_indices).squeeze(1)

        if not self.training and self.normalize_reward:
            # Buffers (mean/std) follow the model when offloaded to CPU; the
            # forward output stays on GPU. Co-locate before the lerp so we
            # don't hit 'Expected all tensors to be on the same device'.
            mean = self.mean.to(reward.device)
            std = self.std.to(reward.device)
            reward = (reward - mean) / std

        return (reward, outputs) if return_output else reward


class CriticModel(_SequenceRegressionBase):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        values_allgather: bool = False,
        cp_local_values: bool = False,
        cp_context_stack=None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        values, outputs = self._token_values(
            input_ids,
            attention_mask,
            cp_local_values=cp_local_values,
            cp_context_stack=cp_context_stack,
        )

        if action_mask is None:
            assert return_output
            return outputs

        if not cp_local_values:
            values = values[:, :-1]
        if self.normalize_reward:
            mean = self.mean.to(values.device)
            std = self.std.to(values.device)
            values = (values - mean) / std

        selected_values = values if cp_local_values else values[:, -action_mask.shape[1] :]
        action_values = torch.where(action_mask.bool(), selected_values, torch.zeros_like(selected_values))
        return (action_values, outputs) if return_output else action_values
