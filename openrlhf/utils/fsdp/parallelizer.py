# Copyright (c) 2025, OpenRLHF Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tensor Parallelism utilities using PyTorch native parallelize_module API.

参考 NeMo Automodel 实现，使用 PyTorch 原生 TP API 替代 transformers 内置实现：
- 更可靠的训练支持
- 非 TP 参数不受影响（不会有梯度同步问题）
- use_local_output=True 确保激活值以普通 Tensor 传递

使用方法:
    from openrlhf.utils.fsdp.parallelizer import apply_tensor_parallel, get_tp_plan

    # 1. 加载模型（不带 TP）
    model = AutoModelForCausalLM.from_pretrained(...)

    # 2. 获取 TP plan
    tp_plan = get_tp_plan(model)

    # 3. 应用 TP
    apply_tensor_parallel(model, tp_mesh, tp_plan)

    # 4. 应用 FSDP
    model = fully_shard(model, ...)
"""

import logging
import re
from functools import lru_cache
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_torch_tp_available() -> bool:
    """检查 PyTorch 版本是否支持 tensor parallel API。"""
    try:
        from packaging import version

        torch_version = version.parse(torch.__version__.split("+")[0])
        return torch_version >= version.parse("2.5.0")
    except Exception:
        return False


if _is_torch_tp_available():
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )
    from torch.distributed.tensor.placement_types import Replicate, Shard


@lru_cache
def translate_to_torch_parallel_style(style: str) -> "ParallelStyle":
    """
    将字符串描述转换为 PyTorch ParallelStyle 对象。

    Args:
        style: 并行风格字符串，如 "colwise", "rowwise" 等

    Returns:
        对应的 ParallelStyle 对象
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def get_base_tp_plan(sequence_parallel: bool = False) -> Dict[str, "ParallelStyle"]:
    """
    获取通用的 base TP plan，适用于 LLaMA 风格模型。

    Args:
        sequence_parallel: 是否启用 sequence parallel

    Returns:
        TP plan 字典
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    base_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate+up
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Replicate()),
    }

    if sequence_parallel:
        sp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        }
        base_plan.update(sp_plan)

    return base_plan


def get_hf_tp_plan(model: nn.Module) -> Dict[str, "ParallelStyle"]:
    """
    从 HuggingFace 模型获取 TP plan。

    优先级：
    1. model_cls._tp_plan
    2. model._tp_plan
    3. model.model._tp_plan (inner model)

    Args:
        model: HuggingFace 模型

    Returns:
        TP plan 字典
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    model_cls = type(model)
    hf_tp_plan = {}

    # 从 model class 获取
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        hf_tp_plan.update(model_cls._tp_plan)

    # 从 model instance 获取
    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    # 从 inner model 获取
    if hasattr(model, "model") and hasattr(model.model, "_tp_plan") and model.model._tp_plan is not None:
        for k, v in model.model._tp_plan.items():
            hf_tp_plan[f"model.{k}"] = v

    if not hf_tp_plan:
        return {}

    # 添加 embed_tokens（如果缺失）
    if "model.embed_tokens" not in hf_tp_plan:
        hf_tp_plan["model.embed_tokens"] = "rowwise_rep"

    # 转换字符串为 ParallelStyle 对象
    result = {}
    for k, v in hf_tp_plan.items():
        if isinstance(v, str):
            try:
                result[k] = translate_to_torch_parallel_style(v)
            except ValueError as e:
                logger.warning(f"Skipping unknown parallel style for {k}: {e}")
        else:
            result[k] = v

    # 为训练稳定性，强制 lm_head 输出为 Replicate（保证 logits 是全量 Tensor）
    if "lm_head" in result and isinstance(result["lm_head"], ColwiseParallel):
        result["lm_head"] = ColwiseParallel(output_layouts=Replicate())

    logger.info(f"[parallelizer] HF TP plan: {list(result.keys())}")
    return result


def get_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    use_hf_tp_plan: bool = True,
    custom_plan: Optional[Dict[str, Union[str, "ParallelStyle"]]] = None,
) -> Dict[str, "ParallelStyle"]:
    """
    获取模型的 TP plan。

    优先级：
    1. custom_plan（如果提供）
    2. HF 模型的 _tp_plan（如果 use_hf_tp_plan=True）
    3. 通用 base plan

    Args:
        model: 模型
        sequence_parallel: 是否启用 sequence parallel
        use_hf_tp_plan: 是否尝试从 HF 模型获取 TP plan
        custom_plan: 自定义 TP plan

    Returns:
        TP plan 字典
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    # 1. 使用自定义 plan
    if custom_plan is not None:
        result = {}
        for k, v in custom_plan.items():
            if isinstance(v, str):
                result[k] = translate_to_torch_parallel_style(v)
            else:
                result[k] = v
        logger.info("[parallelizer] Using custom TP plan")
        return result

    # 2. 尝试从 HF 获取
    if use_hf_tp_plan:
        hf_plan = get_hf_tp_plan(model)
        if hf_plan:
            logger.info("[parallelizer] Using HF TP plan")
            return hf_plan

    # 3. 使用 base plan
    logger.info("[parallelizer] Using base TP plan (LLaMA-style)")
    return get_base_tp_plan(sequence_parallel)


def validate_tp_mesh(model: nn.Module, tp_mesh: "DeviceMesh") -> None:
    """
    验证 TP mesh 与模型的兼容性。

    检查 attention heads 是否可被 TP size 整除。

    Args:
        model: 模型
        tp_mesh: TP device mesh

    Raises:
        AssertionError: 如果不兼容
    """
    if tp_mesh.size() == 1:
        return

    config = getattr(model, "config", None)
    if config is None:
        logger.warning("[parallelizer] Model has no config, skipping TP validation")
        return

    num_attention_heads = getattr(config, "num_attention_heads", 0)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

    if num_attention_heads > 0:
        assert num_attention_heads % tp_mesh.size() == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by " f"TP size ({tp_mesh.size()})"
        )

    if num_key_value_heads > 0:
        assert num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads ({num_key_value_heads}) must be divisible by " f"TP size ({tp_mesh.size()})"
        )

    logger.info(
        f"[parallelizer] TP validation passed: heads={num_attention_heads}, "
        f"kv_heads={num_key_value_heads}, tp_size={tp_mesh.size()}"
    )


def _expand_wildcard_plan(plan: Dict[str, "ParallelStyle"], model: nn.Module) -> Dict[str, "ParallelStyle"]:
    """
    展开 plan 中的通配符（*.）为具体的模块名。

    Args:
        plan: 包含通配符的 TP plan
        model: 模型

    Returns:
        展开后的 TP plan
    """
    expanded = {}
    module_names = {name for name, _ in model.named_modules()}

    for pattern, style in plan.items():
        if "*" not in pattern:
            # 无通配符，直接使用
            if pattern in module_names or pattern == "":
                expanded[pattern] = style
            continue

        # 将 pattern 转换为正则表达式
        # "model.layers.*.self_attn.q_proj" -> "model\.layers\.\d+\.self_attn\.q_proj"
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"\d+")
        regex = re.compile(f"^{regex_pattern}$")

        for name in module_names:
            if regex.match(name):
                expanded[name] = style

    logger.debug(f"[parallelizer] Expanded plan: {len(plan)} patterns -> {len(expanded)} modules")
    return expanded


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: "DeviceMesh",
    tp_plan: Optional[Dict[str, "ParallelStyle"]] = None,
    sequence_parallel: bool = False,
    validate: bool = True,
) -> nn.Module:
    """
    应用 Tensor Parallelism 到模型。

    使用 PyTorch 原生 parallelize_module API。

    Args:
        model: 模型
        tp_mesh: TP device mesh
        tp_plan: TP plan，如果为 None 则自动生成
        sequence_parallel: 是否启用 sequence parallel
        validate: 是否验证 TP 兼容性

    Returns:
        应用了 TP 的模型
    """
    if not _is_torch_tp_available():
        raise RuntimeError("Tensor parallel requires PyTorch >= 2.5.0")

    if tp_mesh.size() == 1:
        logger.info("[parallelizer] TP size is 1, skipping tensor parallel")
        return model

    # 验证兼容性
    if validate:
        validate_tp_mesh(model, tp_mesh)

    # 获取 TP plan
    if tp_plan is None:
        tp_plan = get_tp_plan(model, sequence_parallel=sequence_parallel)

    if not tp_plan:
        logger.warning("[parallelizer] Empty TP plan, skipping tensor parallel")
        return model

    # 展开通配符
    expanded_plan = _expand_wildcard_plan(tp_plan, model)

    if not expanded_plan:
        logger.warning("[parallelizer] No modules matched TP plan patterns")
        return model

    logger.info(f"[parallelizer] Applying TP to {len(expanded_plan)} modules")

    # 应用 TP
    parallelize_module(model, tp_mesh, expanded_plan)

    return model
