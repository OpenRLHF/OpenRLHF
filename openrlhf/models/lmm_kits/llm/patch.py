from ..base.patch import BasePatch
import torch

class LLMPatch(BasePatch):
    '''
    This patch is used to hack the common LLM models for compatibility with LMMs.
    '''
    def _add_get_inputs_embeds():
        from transformers.modeling_utils import PreTrainedModel
        def get_inputs_embeds(self, input_ids, **kwargs):
            if input_ids is None:
                return None
            return self.get_input_embeddings()(input_ids)
        PreTrainedModel.get_inputs_embeds = get_inputs_embeds
    
    def _add_get_position_ids():
        from transformers.modeling_utils import PreTrainedModel
        def get_position_ids(self, input_ids, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        PreTrainedModel.get_position_ids = get_position_ids
    
    def _add_offset_split_position_ids():
        from transformers.modeling_utils import PreTrainedModel
        def offset_split_position_ids(self, split_position_ids, hacked_position_ids):
            # For common position_ids, hacked_position_ids is what we want
            return split_position_ids
        PreTrainedModel.offset_split_position_ids = offset_split_position_ids
    
    def apply_liger_kernel():
        # For LLM, we directly apply liger_kernel in get_generation_cls
        pass
    
    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()

Patch = LLMPatch()