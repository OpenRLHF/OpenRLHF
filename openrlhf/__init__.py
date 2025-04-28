import importlib


ACCELERATOR_TYPE = "GPU"

if importlib.util.find_spec("torch_npu"):
    ACCELERATOR_TYPE = "NPU"

    import torch_npu # noqa
    from torch_npu.contrib import transfer_to_npu # noqa
