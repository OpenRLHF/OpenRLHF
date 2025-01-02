from transformers import is_torch_npu_available

if is_torch_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
