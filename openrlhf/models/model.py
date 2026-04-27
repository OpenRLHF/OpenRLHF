from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, BitsAndBytesConfig

from openrlhf.utils.fsdp.packing import is_automodel_custom_model, pack_padded_batch, unpack_to_padded, unshard_dtensor
from openrlhf.utils.logging_utils import init_logger

from .actor import _build_peft_config_dict, _hf_packing_supported, _will_use_hf_model

logger = init_logger(__name__)


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    param_dtype: str = "bf16",
    load_in_4bit: bool = False,
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
    use_liger_kernel: bool = False,
    use_fp32_master_weights: Optional[bool] = None,
    **kwargs,
) -> nn.Module:
    """Build a reward or critic model with an AutoModel-managed regression head."""
    assert model_type in ("critic", "reward"), f"invalid model_type: {model_type}"

    if packing_samples and model_type == "reward":
        raise NotImplementedError("AutoModel reward models do not support --fsdp.packing_samples")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.num_labels = 1
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    config.value_head_prefix = value_head_prefix
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    from openrlhf.utils.utils import convert_to_torch_dtype, ensure_torchvision_nms_stub

    compute_dtype = convert_to_torch_dtype(param_dtype)
    if use_fp32_master_weights is None:
        use_fp32_master_weights = model_type != "reward"
    torch_dtype = compute_dtype if load_in_4bit or not use_fp32_master_weights else torch.float32

    nf4_config = None
    if load_in_4bit:
        assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    peft_config = None
    if lora_rank > 0:
        peft_config = _build_peft_config_dict(lora_rank, lora_alpha, lora_dropout, target_modules)

    ensure_torchvision_nms_stub()
    from nemo_automodel import NeMoAutoModelForSequenceClassification

    # AutoModel's custom registry is CausalLM-oriented. When a base config says
    # e.g. "LlamaForCausalLM", NeMoAutoModelForSequenceClassification would
    # otherwise instantiate the custom causal model and no regression head would
    # exist. Force the HF SequenceClassification path to preserve OpenRLHF's
    # reward/critic semantics.
    force_hf = True
    automodel_has_packed_sequence = packing_samples
    if (
        packing_samples
        and _will_use_hf_model(model_name_or_path, force_hf)
        and not _hf_packing_supported(attn_implementation)
    ):
        print(
            "[Packing] HF fallback sequence-regression packing requires flash_attention_2 + flash_attn. "
            "Disabling packing for this model and using padded forward instead."
        )
        automodel_has_packed_sequence = False
        packing_samples = False
        if attn_implementation == "flash_attention_2":
            attn_implementation = "sdpa"
            config._attn_implementation = attn_implementation

    model = NeMoAutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=nf4_config,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        moe_config=moe_config,
        activation_checkpointing=activation_checkpointing,
        peft_config=peft_config,
        use_liger_kernel=use_liger_kernel,
        has_packed_sequence=automodel_has_packed_sequence,
        force_hf=force_hf,
        **kwargs,
    )

    if "output_router_logits" in model.config.to_dict():
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
    model.config.use_cache = False
    model.config.normalize_reward = normalize_reward
    model.config.value_head_prefix = value_head_prefix

    if init_value_head:
        _init_regression_head(model, value_head_prefix)

    wrapper_cls = RewardModel if model_type == "reward" else CriticModel
    return wrapper_cls(model, value_head_prefix, packing_samples, normalize_reward)


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
        packing_samples: bool,
        normalize_reward: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.value_head_prefix = value_head_prefix
        self.packing_samples = packing_samples
        self.normalize_reward = normalize_reward
        self._packing_style = "automodel" if is_automodel_custom_model(model) else "hf"
        if self.packing_samples:
            path = "AutoModel THD" if self._packing_style == "automodel" else "HF FlashAttention varlen"
            print(f"[Packing] Sequence regression using {path} packed path.")

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

    def _prepare_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, seqlen = input_ids.size()
        fa_kwargs: dict = {}
        indices = None
        if self.packing_samples:
            input_ids, position_ids, _, indices, fa_kwargs = pack_padded_batch(
                input_ids, attention_mask, style=self._packing_style
            )
            forward_attention_mask = None
        else:
            forward_attention_mask = attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        return batch, seqlen, input_ids, forward_attention_mask, position_ids, indices, fa_kwargs

    def _token_values(self, input_ids, attention_mask):
        batch, seqlen, input_ids, forward_attention_mask, position_ids, indices, fa_kwargs = self._prepare_inputs(
            input_ids, attention_mask
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=forward_attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **fa_kwargs,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Sequence regression requires hidden_states; model did not return them.")
        last_hidden_states = unshard_dtensor(hidden_states[-1])
        values = _get_regression_head(self.model, self.value_head_prefix)(last_hidden_states).squeeze(-1).float()
        if self.packing_samples:
            values = unpack_to_padded(values, indices, batch, seqlen)
        return values, outputs


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
        packed_seq_lens=None,
    ) -> torch.Tensor:
        values, outputs = self._token_values(input_ids, attention_mask)

        if action_mask is None:
            assert return_output
            return outputs

        values = values[:, :-1]
        if self.normalize_reward:
            mean = self.mean.to(values.device)
            std = self.std.to(values.device)
            values = (values - mean) / std

        action_values = values[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_values, outputs) if return_output else action_values
