from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16 if bf16 else "auto"
    )
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        **kwargs,
    )

    return model


def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceClassification(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(self.model.config.hidden_size, 1)
            self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.mean = nn.Parameter(data=torch.zeros(1), requires_grad=False)
            self.std = nn.Parameter(data=torch.ones(1), requires_grad=False)

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            return reward

    return LLMForSequenceClassification


def _get_critic_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceClassification(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(self.model.config.hidden_size, 1)
            self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.mean = nn.Parameter(data=torch.zeros(1), requires_grad=False)
            self.std = nn.Parameter(data=torch.ones(1), requires_grad=False)

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1]
            num_actions = action_mask.size(1)

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std
            return values[:, -num_actions:]

    return LLMForSequenceClassification
