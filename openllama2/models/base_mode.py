import abc
from abc import ABC
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from .utils import log_probs_from_logits


def lora_target_module(target_module="attn"):
    if target_module == "qv":
        return ["q_proj", "v_proj"]
    elif target_module == "attn":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]


class Base_Model(nn.Module, ABC):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        from_config=False,
        lora_args=None,
        compute_dtype=torch.float32,
        gradient_checkpointing=False,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        super().__init__()
        self.pretrain_or_model = pretrain_or_model
        self.compute_dtype = compute_dtype
        self.gradient_checkpointing = gradient_checkpointing
        if lora_args is not None:
            self.qlora = lora_args.q_lora

        if self.qlora:
            self.qlora_init()
        else:
            self.init_model(from_config)

    def init_model(self, from_config=False):
        if isinstance(self.pretrain_or_model, str):
            if from_config:
                config = AutoConfig.from_pretrained(self.pretrain_or_model, torch_dtype="auto")
                self.model = AutoModelForCausalLM.from_config(config)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.pretrain_or_model,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
        else:
            self.model = self.pretrain_or_model

    def qlora_init(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrain_or_model, torch_dtype="auto", trust_remote_code=True, quantization_config=bnb_config
        )
        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=self.gradient_checkpointing
        )

    @abc.abstractmethod
    def generate(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        pass

    @abc.abstractmethod
    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        pass

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def lora_enable(self, lora_args):
        # if lora_rank > 0:
        #     lora_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         inference_mode=False,
        #         r=lora_rank,
        #         lora_alpha=16,
        #         lora_dropout=0.05,
        #         bias=lora_train_bias,
        #     )
        #     self.model = get_peft_model(self.model, lora_config)
        target_modules = lora_target_module(lora_args.lora_target_module)
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            inference_mode=False,
            lora_alpha=lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
