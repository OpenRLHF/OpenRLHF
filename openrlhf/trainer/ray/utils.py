import os


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


#######################
# deepspeed offload
#######################
from deepspeed.runtime.engine import DeepSpeedEngine


def _get_deepspeed_engine(ctx) -> DeepSpeedEngine:
    from openrlhf.models import Actor

    model = ctx.model.model if isinstance(ctx.model, Actor) else ctx.model
    assert isinstance(model, DeepSpeedEngine), "Model must be a DeepSpeedEngine instance"
    
    return model


def offload_deepspeed_states(ctx, pin_memory=True, non_blocking=True):
    assert ctx.model is not None

    model = _get_deepspeed_engine(ctx)
    zero_stage = model.zero_optimization_stage() # config['zero_optimization']['stage']
    adam_offload = model.config['zero_optimization']['offload_optimizer']['device'] == 'cpu'

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum, OffloadDeviceEnum

    model.optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
            # OffloadStateTypeEnum.lp_grads,
            # OffloadStateTypeEnum.lp_params, # dangerous
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.synchronize()


def reload_deepspeed_states(ctx, non_blocking=True):
    assert ctx.model is not None
    
    model = _get_deepspeed_engine(ctx)
    zero_stage = model.zero_optimization_stage() # config['zero_optimization']['stage']
    adam_offload = model.config['zero_optimization']['offload_optimizer']['device'] == 'cpu'
    
    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    ctx.model.reload_states(non_blocking=non_blocking)
    torch.cuda.synchronize()
