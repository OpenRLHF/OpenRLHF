# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

import deepspeed
import numpy as np
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    enable_hybrid_engine=False,
    inference_tp_size=1,
    release_inference_cache=True,
    pin_parameters=True,
    tp_gather_partition_size=4,
    max_out_tokens=512,
    zpg=8,
):
    if enable_hybrid_engine and inference_tp_size > 1:
        assert stage == 3, "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
    enable_hybrid_engine=False,
    inference_tp_size=1,
    release_inference_cache=True,
    pin_parameters=True,
    tp_gather_partition_size=1,
    max_out_tokens=512,
):
    if enable_hybrid_engine and inference_tp_size > 1:
        assert stage == 3, "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": "cpu" if offload else "none",
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
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
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
