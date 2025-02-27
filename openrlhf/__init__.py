import importlib


def is_torch_npu_available():
    return importlib.util.find_spec("torch_npu")

IS_NPU_AVAILABLE = is_torch_npu_available()

ACCELERATOR_TYPE = "GPU"

if IS_NPU_AVAILABLE:
    ACCELERATOR_TYPE = "NPU"

    import torch_npu
    from torch_npu.contrib import transfer_to_npu
