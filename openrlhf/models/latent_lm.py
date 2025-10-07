from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


class LatentLM(nn.Module):
    """
    Base class for LatentLM models in latent language modeling.

    This class serves as a foundation for implementing various latent language models, which are responsible for generating latent representations of language.
    Example of latent token: <LATENT_0>, <LATENT_1>, <LATENT_2>, ...

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the latent language model.
        codebook_size (int): Size of the codebook.
        tokenizer (Tokenizer, optional): Tokenizer to use. Defaults to None.
        padding_side (str, optional): Padding side for the tokenizer. Defaults to "right".
        init_method (str, optional): Method to initialize the new embeddings. Defaults to "mean".
        use_fast (bool, optional): Whether to use fast tokenizer. Defaults to True.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for latent language modeling. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        codebook_size,
        strategy,
        tokenizer=None,
        padding_side="right",
        init_method="mean",
        use_fast=True,
        attn_implementation="flash_attention_2",
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.codebook_size = codebook_size
        self.strategy = strategy

        if isinstance(pretrain_or_model, str):
            # Support multiple attention mechanism implementations
            attn_impl = attn_implementation

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrain_or_model,trust_remote_code=True, use_fast=use_fast)
                self.tokenizer.padding_side = padding_side
                # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
                # https://github.com/facebookresearch/llama-recipes/pull/196
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    if self.model is not None:
                        self.model.config.pad_token_id = self.tokenizer.pad_token_id

            



            

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                raise NotImplementedError("MoE is not supported yet")
                self.strategy.print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

                # set_z3_leaf_modules is required for MoE models
                for m in self.model.modules():
                    # https://github.com/microsoft/DeepSpeed/pull/4966
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        self.strategy.print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                        break

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

        ## Extend vocab size to handle latent tokens

        # get latent token list
        latent_token_list = []
        for i in range(codebook_size):
            latent_token_list.append(f"<LATENT_{i}>")
        
        # add latent tokens to tokenizer
        n_added = self.tokenizer.add_tokens(latent_token_list)
        self.strategy.print(f"tokenizer.add_tokens returned: {n_added}")

        # get input embeddings
        input_emb = self.model.get_input_embeddings()   # nn.Embedding
        old_num_tokens, emb_dim = input_emb.weight.size()
        
        # resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))  # maybe tie_weights is handled internally
        new_input_emb = self.model.get_input_embeddings()
        new_num_tokens = new_input_emb.weight.size(0)
        self.strategy.print(f"new embedding shape: {new_num_tokens} x {self.model.config.hidden_size} (expected {self.model.config.vocab_size + n_added})")


        # initialize new embeddings
        with torch.no_grad():
            if init_method == "mean":
                # initialize new embeddings with the mean of existing embeddings
                mean_vec = new_input_emb.weight.mean(dim=0, keepdim=True)  # 1 x D
                new_input_emb.weight.data[self.model.config.vocab_size:] = mean_vec.repeat(n_added, 1)
                self.strategy.print("Initialized new embeddings with the mean of existing embeddings.")

            
            elif init_method == "random":
                # initialize new embeddings randomly
                std = getattr(self.model.config, "initializer_range", 0.02)
                new_input_emb.weight.data[old_num_tokens:] = torch.randn(n_added, emb_dim) * std
                self.strategy.print("Initialized new embeddings randomly (normal with initializer_range).")

            else:
                raise ValueError("init_method must be one of 'mean','random'")

        # tie_weights: maybe tied internally, but we call it explicitly
        try:
            self.model.tie_weights()
            self.strategy.print("Called model.tie_weights() to ensure input/output embeddings are tied (if supported).")
        except Exception as e:
            self.strategy.print("model.tie_weights() not available or failed:", e)


        # LoRA
        if lora_rank > 0:
            # https://github.com/huggingface/peft/issues/137
            self.model.enable_input_require_grads()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

            if load_in_4bit:
                for name, module in self.model.named_modules():
                    if isinstance(module, LoraLayer):
                        module = module.to(torch.bfloat16)
                    if "norm" in name:
                        module = module.to(torch.float32)
                    if "lm_head" in name or "embed_tokens" in name:
                        if hasattr(module, "weight"):
                            module = module.to(torch.bfloat16)

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
