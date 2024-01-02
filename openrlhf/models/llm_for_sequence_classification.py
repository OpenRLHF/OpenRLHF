import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_classification(
    model_name_or_path: str,
    model_type: str,
    num_labels=1,
    normalize_reward=False,
    use_flash_attention_2=False,
    **kwargs,
):
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): _description_
        model_type (str): 'critic' or 'reward'.
        config (_type_, optional): _description_. Defaults to None.
        num_labels (int, optional): _description_. Defaults to 1.
        use_flash_attention_2 (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    if config is None:
        config, kwargs = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, return_unused_kwargs=True, trust_remote_code=True
        )

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

        print(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path, **kwargs
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path, **kwargs)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)

    config.normalize_reward = normalize_reward

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=use_flash_attention_2,
    )

    return model


def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceClassification(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.num_labels = config.num_labels
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(self.model.config.hidden_size, 1)
            self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))

            # Initialize weights and apply final processing
            self.post_init()

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

        def lora_enable(self, lora_rank=0, lora_train_bias="none"):
            if lora_rank > 0:
                lora_config = LoraConfig(
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias=lora_train_bias,
                )
                self.model = get_peft_model(self.model, lora_config)

    return LLMForSequenceClassification


def _get_critic_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceClassification(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.num_labels = config.num_labels
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(self.model.config.hidden_size, 1)
            self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            action_mask: Optional[torch.Tensor] = None,
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

        def lora_enable(self, lora_rank=0, lora_train_bias="none"):
            if lora_rank > 0:
                lora_config = LoraConfig(
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias=lora_train_bias,
                )
                self.model = get_peft_model(self.model, lora_config)

    return LLMForSequenceClassification
