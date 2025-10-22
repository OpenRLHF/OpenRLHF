import json
import os
import re
from typing import Dict, List, Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


def unicode_range(start, end):
    return "".join(chr(c) for c in range(start, end + 1))


VALID_LATEXY = re.compile(
    r"\A["
    + r"A-Za-z0-9"
    + r"\s"
    + r"\.\,\;\:\!\?\'\"\_\-\+\=\*\/\^\|\%\<\>\~\#\@\&\$"
    + re.escape("\\()[]{}")  # 특수문자 안전하게
    + unicode_range(0x0370, 0x03FF)  # Greek and Coptic
    + unicode_range(0x1F00, 0x1FFF)  # Greek Extended
    + unicode_range(0x2190, 0x21FF)  # Arrows
    + unicode_range(0x2200, 0x22FF)  # Math Operators
    + unicode_range(0x27C0, 0x27EF)  # Misc Math A
    + unicode_range(0x2980, 0x29FF)  # Misc Math B
    + unicode_range(0x2A00, 0x2AFF)  # Supplemental Math
    + unicode_range(0x2100, 0x214F)  # Letterlike symbols
    + unicode_range(0x2070, 0x209F)  # Superscripts/Subscripts
    + unicode_range(0x1D400, 0x1D7FF)  # Math alphanumeric
    + "\u00b0\u00b1\u00b2\u00b3\u00b7\u00b9\u00d7\u00f7"
    + r"]*\Z",
    re.UNICODE,
)


def is_valid_generated_text(text: str) -> bool:
    return bool(VALID_LATEXY.match(text))


def count_char_len(s: str) -> int:
    return len(s)


def _pick_invalid_tokens(vocab: Dict[str, int], need: int) -> List[str]:
    """
    pick invalid tokens from vocab
    """
    # token, id, cyr_count
    items = [(tok, tid, count_char_len(tok)) for tok, tid in vocab.items()]
    # priority: cyr_count desc, token length desc(more stable for candidates with 3 or more characters), id asc
    items.sort(key=lambda x: (-x[2], -len(x[0]), x[1]))

    def take(taken: set) -> List[str]:
        out = []
        for tok, _, c in items:
            if tok in taken:
                continue
            if not is_valid_generated_text(tok):
                out.append(tok)
                if len(out) + len(taken) >= need:
                    break
        return out

    chosen = []
    seen = set()

    batch = take(seen)
    chosen += batch
    seen.update(batch)

    if len(chosen) >= need:
        return chosen[:need]
    else:
        raise ValueError("Not enough invalid tokens to replace")


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
        start_cp=0x0410,
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
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrain_or_model, trust_remote_code=True, use_fast=use_fast
                )
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

        # Replace unused tokens to Latent tokens
        # Rebuild tokenizer and initialize embeddings
        self.tokenizer, self.model, _, _, _ = self.rebuild_tokenizer_by_invalid_tokens(
            self.model,
            self.tokenizer,
            strategy.args.save_path,
            None,
            codebook_size,
            token_prefix="<LATENT_",
            verbose=True,
            init_embeddings=True,
            init_method=init_method,
            torch_dtype=torch.float32,
        )

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

    def rebuild_tokenizer_by_invalid_tokens(
        self,
        model,
        tokenizer,
        save_path: str,
        model_path: str = None,
        codebook_size: int = 1024,
        token_prefix: str = "<LATENT_",
        verbose: bool = True,
        init_embeddings: bool = True,
        init_method: str = "mean",
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        1) load original tokenizer & save
        2) pick invalid tokens from vocab
        3) replace selected tokens with <LATENT_i>
        4) remove merge rules involving replaced tokens
        5) add new tokens to added_tokens / added_tokens_decoder
        """
        # set rank (for distributed environment)
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            os.makedirs(save_path, exist_ok=True)
            if tokenizer is not None:
                tok = tokenizer

            else:
                if verbose:
                    print(f"🔹 Loading original tokenizer from: {model_path}")
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

            if verbose:
                print(f"🔹 Saving tokenizer to: {save_path}")
            tok.save_pretrained(save_path)

            # file paths
            tokjson = os.path.join(save_path, "tokenizer.json")
            tokcfg = os.path.join(save_path, "tokenizer_config.json")
            vocabjson = os.path.join(save_path, "vocab.json")
            mergestxt = os.path.join(save_path, "merges.txt")
            addedtoks = os.path.join(save_path, "added_tokens.json")

            # load
            tokdata = None
            vocab = None
            merges_json = None
            if os.path.exists(tokjson):
                with open(tokjson, "r", encoding="utf-8") as f:
                    tokdata = json.load(f)
                vocab = tokdata["model"]["vocab"]
                merges_json = tokdata["model"].get("merges", [])

            vocab_json_file = None
            if os.path.exists(vocabjson):
                with open(vocabjson, "r", encoding="utf-8") as f:
                    vocab_json_file = json.load(f)
                    if vocab is None:
                        vocab = vocab_json_file

            merges_txt = None
            if os.path.exists(mergestxt):
                with open(mergestxt, "r", encoding="utf-8") as f:
                    merges_txt = f.read()

            added_tokens = None
            if os.path.exists(addedtoks):
                with open(addedtoks, "r", encoding="utf-8") as f:
                    added_tokens = json.load(f)

            # pick candidates
            candidates = _pick_invalid_tokens(vocab, codebook_size)
            new_tokens = [f"{token_prefix}{i}>" for i in range(codebook_size)]
            mapping = dict(zip(candidates, new_tokens))

            # vocab rename
            replaced_ids, new_tok_strs = [], []
            for old, new in mapping.items():
                if old in vocab:
                    tid = vocab.pop(old)
                    vocab[new] = tid
                    replaced_ids.append(tid)
                    new_tok_strs.append(new)
                if vocab_json_file and old in vocab_json_file:
                    tid2 = vocab_json_file.pop(old)
                    vocab_json_file[new] = tid2

            # modify merges → remove merge rules involving replaced tokens
            removed_rules = []
            kept_merges_json = []
            removed_rules_txt = []
            kept_merges_txt = []
            if merges_json is not None:
                for rule in merges_json:
                    merged = "".join(rule)
                    if any(old in merged for old in mapping.keys()):
                        removed_rules.append(rule)
                    else:
                        kept_merges_json.append(rule)
            if merges_txt is not None:
                for rule in merges_txt.split("\n"):
                    splited_rule = rule.split()
                    merged = "".join(splited_rule)
                    if any(old in merged for old in mapping.keys()):
                        removed_rules_txt.append(rule)
                    else:
                        kept_merges_txt.append(rule)
            if verbose:
                print(f"🧹 Removed {len(removed_rules)} merge rules involving replaced tokens.")

            # added_tokens / added_tokens_decoder add
            added_specs = list(zip(replaced_ids, new_tok_strs))

            # tokenizer_config.json modify
            if os.path.exists(tokcfg):
                with open(tokcfg, "r", encoding="utf-8") as f:
                    tokcfg_data = json.load(f)
            else:
                tokcfg_data = {}

            decoder = tokcfg_data.get("added_tokens_decoder", {})
            if not isinstance(decoder, dict):
                decoder = {}

            for tid, content in added_specs:
                decoder[str(int(tid))] = {
                    "content": content,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": True,
                    "special": False,
                }

            tokcfg_data["added_tokens_decoder"] = decoder
            with open(tokcfg, "w", encoding="utf-8") as f:
                json.dump(tokcfg_data, f, ensure_ascii=False, indent=2)

            # tokenizer.json modify
            if tokdata is not None:
                added_list = tokdata.get("added_tokens", [])

                if not isinstance(added_list, list):
                    added_list = []

                id_to_idx = {int(e["id"]): i for i, e in enumerate(added_list) if "id" in e}
                for tid, content in added_specs:
                    payload = {
                        "id": int(tid),
                        "content": content,
                        "single_word": True,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": False,
                    }
                    if tid in id_to_idx:
                        added_list[id_to_idx[tid]] = payload
                    else:
                        added_list.append(payload)

                tokdata["added_tokens"] = added_list
                tokdata["model"]["vocab"] = vocab
                tokdata["model"]["merges"] = kept_merges_json

            else:
                for tid, content in added_specs:
                    added_tokens[content] = tid

            # save
            if tokdata:
                with open(tokjson, "w", encoding="utf-8") as f:
                    json.dump(tokdata, f, ensure_ascii=False, indent=2)
            if vocab_json_file:
                with open(vocabjson, "w", encoding="utf-8") as f:
                    json.dump(vocab_json_file, f, ensure_ascii=False, indent=2)
            if merges_txt:
                with open(mergestxt, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept_merges_txt))
            if added_tokens:
                with open(addedtoks, "w", encoding="utf-8") as f:
                    json.dump(added_tokens, f, ensure_ascii=False, indent=2)

            if verbose:
                print(f"✅ Saved modified tokenizer with {len(replaced_ids)} replaced tokens.")
                print(f"🔹 Example: {candidates[0]} → {new_tokens[0]}")

            # fast tokenizer load test
            tok2 = AutoTokenizer.from_pretrained(
                save_path,
                local_files_only=True,
                # okenizer_file=os.path.join(save_path, "tokenizer.json"),
                trust_remote_code=False,
                use_fast=True,
            )
            tid = tok2.convert_tokens_to_ids(new_tokens[0])
            dec = tok2.convert_ids_to_tokens([tid])[0]
            if verbose:
                print(f"🔹 Decode check: {new_tokens[0]} → {tid} → {dec}")

            if verbose:
                print(f"✅ Replaced {len(replaced_ids)} tokens. Example: {candidates[0]} → {new_tokens[0]}")
                print(f"🔹 New token id check: {new_tokens[0]} → {tok2.convert_tokens_to_ids(new_tokens[0])}")
                print(
                    f"New token decode check: {tok2.convert_tokens_to_ids(new_tokens[0])} → {tok2.convert_ids_to_tokens(tok2.convert_tokens_to_ids(new_tokens[0]))}"
                )

            # 7) (optional) initialize embeddings/LM-Head
            if init_embeddings:

                if model is None:
                    if verbose:
                        print(f"🔹 Loading model for embedding init from: {model_path}")
                    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)

                emb = model.get_input_embeddings()
                out = model.get_output_embeddings()
                device = emb.weight.device
                emb_dim = emb.weight.shape[1]

                with torch.no_grad():
                    if init_method == "mean":
                        mean_vec = emb.weight.data.mean(dim=0)
                        for tid in replaced_ids:
                            emb.weight.data[tid, :] = mean_vec
                            if out is not None and out is not emb:
                                out.weight.data[tid, :] = mean_vec
                    elif init_method == "random":
                        std = getattr(model.config, "initializer_range", 0.02)
                        noise = torch.randn(len(replaced_ids), emb_dim, device=device) * std
                        for i, tid in enumerate(replaced_ids):
                            emb.weight.data[tid, :] = noise[i]
                            if out is not None and out is not emb:
                                out.weight.data[tid, :] = noise[i]
                    else:
                        raise ValueError("init_method must be 'random' or 'mean'")
                    try:
                        model.tie_weights()
                    except Exception:
                        pass

                # # save modified model is optional
                # if verbose:
                #     print("✅ Embedding/LM-Head re-init done (not saved). Save with model.save_pretrained() if needed.")
        if dist.is_initialized():
            dist.barrier()
        return tok2, model, replaced_ids, new_tokens, candidates

    # def replace_unused_tokens(self, codebook_size, start_cp, init_method="random", save_dir="./custom_tokenizer"):
    #     """
    #     Replace unused tokens (e.g. invalid tokens) with <LATENT_i> custom tokens,
    #     and reinitialize embeddings for these tokens.
    #     """

    #     # set rank (for distributed environment)
    #     rank = dist.get_rank() if dist.is_initialized() else 0

    #     if rank == 0:
    #         # 1️⃣ create custom tokens
    #         latent_token_list = [f"<LATENT_{i}>" for i in range(codebook_size)]
    #         unused_tokens = [chr(c) for c in range(start_cp, start_cp + codebook_size)]
    #         replaced = []
    #         replaced_token_ids = []

    #         # 2️⃣ check fast tokenizer / slow tokenizer
    #         if hasattr(self.tokenizer, "_tokenizer"):  # Fast tokenizer
    #             state = self.tokenizer._tokenizer.model.__getstate__()
    #             vocab = state["vocab"]
    #         else:
    #             vocab = self.tokenizer.vocab

    #         # 3️⃣ replace tokens
    #         for i, old_tok in enumerate(unused_tokens):
    #             if old_tok in vocab and i < len(latent_token_list):
    #                 new_tok = latent_token_list[i]
    #                 tok_id = vocab[old_tok]

    #                 vocab[new_tok] = tok_id
    #                 vocab.pop(old_tok, None)

    #                 replaced.append((old_tok, new_tok))
    #                 replaced_token_ids.append(tok_id)

    #         # if fast tokenizer, reflect __setstate__
    #         if hasattr(self.tokenizer, "_tokenizer"):
    #             self.tokenizer._tokenizer.model.__setstate__(state)

    #         self.strategy.print(f"✅ {len(replaced)} tokens replaced.")

    #         # 4️⃣ initialize embeddings
    #         with torch.no_grad():
    #             emb = self.model.get_input_embeddings()
    #             out_emb = self.model.get_output_embeddings()

    #             device = getattr(self, "device", None) or emb.weight.device
    #             old_weights = emb.weight.data.clone().to(device)
    #             emb_dim = old_weights.size(1)

    #             if init_method == "mean":
    #                 mean_vec = old_weights.mean(dim=0)
    #                 for i, idx in enumerate(replaced_token_ids):
    #                     emb.weight.data[idx, :] = mean_vec
    #                     if out_emb is not None and out_emb is not emb:
    #                         out_emb.weight.data[idx, :] = mean_vec

    #             elif init_method == "random":
    #                 rng = torch.randn(len(replaced_token_ids), emb_dim, device=device) * getattr(
    #                     self.model.config, "initializer_range", 0.02
    #                 )
    #                 for i, idx in enumerate(replaced_token_ids):
    #                     emb.weight.data[idx, :] = rng[i]
    #                     if out_emb is not None and out_emb is not emb:
    #                         out_emb.weight.data[idx, :] = rng[i]
    #             else:
    #                 raise ValueError("init_method must be 'mean' or 'random'")

    #             try:
    #                 self.model.tie_weights()
    #             except Exception:
    #                 pass

    #         # 5️⃣ safe validation
    #         vocab_size = self.model.get_input_embeddings().weight.size(0)
    #         assert max(replaced_token_ids) < vocab_size, "Token ID out of embedding range!"

    #         # 6️⃣ save (tokenizer + mapping)
    #         os.makedirs(save_dir, exist_ok=True)
    #         self.tokenizer.save_pretrained(save_dir)
    #         with open(os.path.join(save_dir, "latent_token_map.json"), "w", encoding="utf-8") as f:
    #             json.dump(dict(replaced), f, ensure_ascii=False, indent=2)

    #         self.strategy.print(f"💾 Saved modified tokenizer & latent map to {save_dir}")

    #     # 7️⃣ (optional) broadcast changes to all ranks in distributed environment
    #     if dist.is_initialized():
    #         dist.barrier()

    #     # 8️⃣ return replaced token ids
    #     return replaced_token_ids


class LatentLM_old(nn.Module):
    """
    ** extend vocab size version--do not use?
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
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrain_or_model, trust_remote_code=True, use_fast=use_fast
                )
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
        input_emb = self.model.get_input_embeddings()  # nn.Embedding
        old_num_tokens, emb_dim = input_emb.weight.size()

        # resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))  # maybe tie_weights is handled internally
        new_input_emb = self.model.get_input_embeddings()
        new_num_tokens = new_input_emb.weight.size(0)
        self.strategy.print(
            f"new embedding shape: {new_num_tokens} x {self.model.config.hidden_size} (expected {self.model.config.vocab_size + n_added})"
        )

        # initialize new embeddings
        with torch.no_grad():
            if init_method == "mean":
                # initialize new embeddings with the mean of existing embeddings
                mean_vec = new_input_emb.weight.mean(dim=0, keepdim=True)  # 1 x D
                new_input_emb.weight.data[self.model.config.vocab_size :] = mean_vec.repeat(n_added, 1)
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
            self.strategy.print(
                "Called model.tie_weights() to ensure input/output embeddings are tied (if supported)."
            )
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
