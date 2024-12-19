from torch._utils import _get_available_device_type, _get_device_module


def get_device_info():
    device_type = _get_available_device_type()

    if device_type is None:
        device_type = "cuda" # default device_type: cuda
    
    device_module = _get_device_module(device_type) # default device_module: torch.cuda

    return device_type, device_module

device_type, device_module = get_device_info()
