RING_ATTN_GROUP = None


def patch_transformers_for_ring_flash_attn():
    """ring_flash_attn<=0.1.8 imports is_flash_attn_greater_or_equal_2_10 from
    transformers.modeling_flash_attention_utils, but transformers>=5.5.4 moved
    it to transformers.utils. Re-export the symbol so the import keeps working.
    See https://github.com/OpenRLHF/OpenRLHF/issues/1222.
    """
    import transformers.modeling_flash_attention_utils as _m

    if not hasattr(_m, "is_flash_attn_greater_or_equal_2_10"):
        from transformers.utils import is_flash_attn_greater_or_equal_2_10

        _m.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


