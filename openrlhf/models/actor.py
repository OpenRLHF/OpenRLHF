import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import set_z3_leaf_modules


# Fix https://github.com/OpenRLHF/OpenRLHF/issues/1232
def _patch_zero3_weight_mapping():
    try:
        import transformers.integrations.deepspeed as hf_deepspeed
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    except (ImportError, AttributeError):
        return

    original_fn = getattr(hf_deepspeed, "_apply_weight_conversions_to_state_dict", None)
    if original_fn is None or getattr(original_fn, "_openrlhf_patched", False):
        return

    def patched_fn(model, state_dict, weight_mapping):
        model_state_dict = model.state_dict()
        renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
        converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
        prefix = model.base_model_prefix

        metadata = getattr(state_dict, "_metadata", None)
        exact_matches = {}
        mapped_state_dict = state_dict.copy()
        if metadata is not None:
            mapped_state_dict._metadata = metadata

        for key in list(state_dict.keys()):
            renamed_key, _ = rename_source_key(key, renamings, converters, prefix, model_state_dict)
            if renamed_key not in model_state_dict and key in model_state_dict:
                exact_matches[key] = mapped_state_dict.pop(key)

        converted_state_dict = original_fn(model, mapped_state_dict, weight_mapping)
        converted_state_dict.update(exact_matches)
        if metadata is not None:
            converted_state_dict._metadata = metadata
        return converted_state_dict

    patched_fn._openrlhf_patched = True
    hf_deepspeed._apply_weight_conversions_to_state_dict = patched_fn


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        param_dtype (str, optional): Model data type ("bf16", "fp16"). Defaults to "bf16".
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        use_liger_kernel=False,
        experts_implementation=None,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            # Support multiple attention mechanism implementations
            attn_impl = attn_implementation

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                _patch_zero3_weight_mapping()
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # Determine torch dtype based on param_dtype parameter, default: bf16
            from openrlhf.utils.utils import convert_to_torch_dtype

            torch_dtype = convert_to_torch_dtype(param_dtype)

            if load_in_4bit:
                assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
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

            extra_from_pretrained_kwargs = {}
            if experts_implementation is not None:
                extra_from_pretrained_kwargs["experts_implementation"] = experts_implementation

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch_dtype,  # default: bf16
                device_map=device_map,
                **extra_from_pretrained_kwargs,
            )

            eff_experts_impl = getattr(
                self.model.config,
                "_experts_implementation_internal",
                getattr(self.model.config, "_experts_implementation", None),
            )
            if eff_experts_impl is not None:
                print(f"[MoE] experts_implementation (resolved): {eff_experts_impl}")

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

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            set_z3_leaf_modules(self.model)

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

        else:
            self.model = pretrain_or_model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
