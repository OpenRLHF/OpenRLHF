from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import set_z3_leaf_modules

logger = init_logger(__name__)


def _init_value_head(model, config):
    """Set up score head and reward normalization buffers on a model instance."""
    model.score = nn.Linear(config.hidden_size, 1, bias=False)
    model.normalize_reward = config.normalize_reward
    model.register_buffer("mean", torch.zeros(1), persistent=False)
    model.register_buffer("std", torch.ones(1), persistent=False)
    if not getattr(model.mean, "is_meta", False):
        model.mean.fill_(float(getattr(config, "mean", 0.0)))
        model.std.fill_(float(getattr(config, "std", 1.0)))


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    torch_dtype: torch.dtype = torch.float32,
    device_map: Optional[str] = "meta",
    normalize_reward=False,
    attn_implementation="flash_attention_2",
    packing_samples=False,
) -> nn.Module:
    """Build a sequence-regression model.

    - ``device_map="meta"`` (default): build model structure on meta device only.
      Weights should be loaded later via ``strategy.load_pretrained``.
    - ``device_map!="meta"``: load real weights via HuggingFace
      ``from_pretrained`` and forward ``device_map`` to it (e.g. ``"cuda:0"``).
    """

    assert model_type in ("critic", "reward"), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation
    config.value_head_prefix = "score"

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, packing_samples)

    if device_map == "meta":
        with torch.device("meta"):
            model = cls_class(config)
    else:
        load_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "dtype": torch_dtype,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        model, loading_info = cls_class.from_pretrained(
            model_name_or_path,
            output_loading_info=True,
            **load_kwargs,
        )
        # Fail-fast: if the checkpoint is missing keys, HF from_pretrained
        # silently random-inits them.  For reward/critic models this means a
        # random score head producing garbage.  Refuse to continue.
        missing = set(loading_info.get("missing_keys", []))
        if missing:
            sample = ", ".join(sorted(missing)[:8])
            raise RuntimeError(
                f"{model_type} checkpoint '{model_name_or_path}' is missing "
                f"required keys: {sample}. Refusing to run with uninitialized "
                f"weights. For FSDP2 training from a base LM, use "
                f"device_map='meta' and load_hf_checkpoint(init_missing_value_head=True)."
            )

    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    set_z3_leaf_modules(model)
    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.packing_samples = packing_samples
            _init_value_head(self, config)
            self.post_init()

        def reset_buffers(self):
            """Refresh GPU buffers from current config values."""
            if not getattr(self.mean, "is_meta", False):
                self.mean.fill_(float(getattr(self.config, "mean", 0.0)))
                self.std.fill_(float(getattr(self.config, "std", 1.0)))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            values = self.score(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.packing_samples = packing_samples
            _init_value_head(self, config)
            self.post_init()

        def reset_buffers(self):
            """Refresh GPU buffers from current config values."""
            if not getattr(self.mean, "is_meta", False):
                self.mean.fill_(float(getattr(self.config, "mean", 0.0)))
                self.std.fill_(float(getattr(self.config, "std", 1.0)))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]
            values = self.score(last_hidden_states).squeeze(-1)  # (1, total_seqs)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            return action_values

    return CriticModel
