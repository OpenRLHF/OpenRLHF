from ..base.patch import BasePatch
import torch

class Phi4MMPatch(BasePatch):
    def _register_to_autoclass():
        from transformers import AutoModelForImageTextToText, AutoConfig, AutoProcessor
        from .src.configuration_phi4mm import Phi4MMConfig
        from .src.modeling_phi4mm import Phi4MMForCausalLM
        from .src.processing_phi4mm import Phi4MMProcessor
        AutoConfig.register("phi4mm", Phi4MMConfig)
        AutoModelForImageTextToText.register(Phi4MMConfig, Phi4MMForCausalLM)
        AutoProcessor.register(Phi4MMConfig, Phi4MMProcessor)
    
    def _add_get_inputs_embeds():
        from .src.modeling_phi4mm import Phi4MMForCausalLM
        def get_inputs_embeds(self, input_ids, num_img_tokens=None,input_mode=None,**kwargs):
            return self.model.embed_tokens_extend(input_ids,wte=self.model.embed_tokens,**kwargs)
        Phi4MMForCausalLM.get_inputs_embeds = get_inputs_embeds
    
    def _add_get_position_ids():
        from .src.modeling_phi4mm import Phi4MMForCausalLM
        def get_position_ids(self, input_ids, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        Phi4MMForCausalLM.get_position_ids = get_position_ids
    
    def _add_offset_split_position_ids():
        from .src.modeling_phi4mm import Phi4MMForCausalLM
        def offset_split_position_ids(self, split_position_ids, hacked_position_ids):
            # For common position_ids, hacked_position_ids is what we want
            return hacked_position_ids
        Phi4MMForCausalLM.offset_split_position_ids = offset_split_position_ids
    
    def _hack_multihead_attention():
        import torch
        import torch.nn as nn
        raw_MultiheadAttention = nn.MultiheadAttention
        """
        MultiheadAttention is not compatible with zero3, it accesses out_proj.weight and out_proj.bias directly, which is partitioned by zero3.
        We re-assign out_proj.weight and out_proj.bias to a new parameter, and the new parameter is not partitioned by zero3.
        """
        class HackedMultiheadAttention(raw_MultiheadAttention):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                output_proj_weight = nn.Parameter(torch.empty((self.embed_dim, self.embed_dim)))
                output_proj_bias = nn.Parameter(torch.zeros(self.embed_dim))
                self.out_proj.weight = output_proj_weight
                self.out_proj.bias = output_proj_bias
        
        torch.nn.MultiheadAttention = HackedMultiheadAttention
                    
    
    def apply_liger_kernel():
        from liger_kernel.transformers import LigerPhi3SwiGLUMLP, LigerRMSNorm, liger_rotary_pos_emb
        from .src import modeling_phi4mm
        modeling_phi4mm.Phi4MMMLP = LigerPhi3SwiGLUMLP
        modeling_phi4mm.Phi4MMRMSNorm = LigerRMSNorm
        modeling_phi4mm.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()
        cls._hack_multihead_attention()
        cls._register_to_autoclass()

Patch = Phi4MMPatch()