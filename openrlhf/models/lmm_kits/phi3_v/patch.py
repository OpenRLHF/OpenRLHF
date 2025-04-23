from ..base.patch import BasePatch
import torch

class Phi3_VPatch(BasePatch):
    def _register_to_autoclass():
        from transformers import AutoModelForImageTextToText, AutoConfig, AutoProcessor
        from .src.configuration_phi3_v import Phi3VConfig
        from .src.modeling_phi3_v import Phi3VForCausalLM
        from .src.processing_phi3_v import Phi3VProcessor
        AutoConfig.register("phi3_v", Phi3VConfig)
        AutoModelForImageTextToText.register(Phi3VConfig, Phi3VForCausalLM)
        AutoProcessor.register(Phi3VConfig, Phi3VProcessor)
    
    def _add_get_inputs_embeds():
        from .src.modeling_phi3_v import Phi3VForCausalLM
        def get_inputs_embeds(self, input_ids, pixel_values=None, image_sizes=None, **kwargs):
            return self.model.vision_embed_tokens(input_ids, pixel_values=pixel_values, image_sizes=image_sizes)
        Phi3VForCausalLM.get_inputs_embeds = get_inputs_embeds
    
    def _add_get_position_ids():
        from .src.modeling_phi3_v import Phi3VForCausalLM
        def get_position_ids(self, input_ids, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        Phi3VForCausalLM.get_position_ids = get_position_ids
    
    def _add_offset_split_position_ids():
        from .src.modeling_phi3_v import Phi3VForCausalLM
        def offset_split_position_ids(self, split_position_ids, hacked_position_ids):
            # For common position_ids, hacked_position_ids is what we want
            return hacked_position_ids
        Phi3VForCausalLM.offset_split_position_ids = offset_split_position_ids
    
    def apply_liger_kernel():
        from liger_kernel.transformers import LigerPhi3SwiGLUMLP, LigerRMSNorm, liger_rotary_pos_emb
        from .src import modeling_phi3_v
        modeling_phi3_v.Phi3MLP = LigerPhi3SwiGLUMLP
        modeling_phi3_v.Phi3RMSNorm = LigerRMSNorm
        modeling_phi3_v.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()
        cls._register_to_autoclass()

Patch = Phi3_VPatch()