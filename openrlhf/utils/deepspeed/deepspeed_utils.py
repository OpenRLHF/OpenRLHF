from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    overlap_comm=False,
    use_ds_universal_ckpt=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
    if stage == 3:
        zero_opt_dict["reduce_scatter"] = True

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype},
        "checkpoint": {
            "load_universal": use_ds_universal_ckpt,
        },
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def offload_deepspeed_states(model, pin_memory=True, non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import deepspeed
    import torch
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    offload_state_types = [
        OffloadStateTypeEnum.optim_states,
        OffloadStateTypeEnum.contiguous_grad_buffer,
        OffloadStateTypeEnum.hp_params,
    ]

    if deepspeed.__version__ >= "0.16.5":
        # These offload types are fixed in https://github.com/deepspeedai/DeepSpeed/pull/7050
        offload_state_types += [
            OffloadStateTypeEnum.lp_grads,
            # OffloadStateTypeEnum.lp_params,
        ]

    model.optimizer.offload_states(
        include=offload_state_types,
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()


def reload_deepspeed_states(model, non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    model.reload_states(non_blocking=non_blocking)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()
