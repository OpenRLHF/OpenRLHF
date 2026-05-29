import deepspeed
import torch.nn as nn


def set_z3_leaf_modules(model: nn.Module) -> None:
    """Auto-detect and set DeepSpeed ZeRO3 leaf modules.

    ZeRO3 prefetches submodule parameters assuming a fixed module traversal order.
    This breaks for:
      - MoE: dynamic expert routing makes prefetch unpredictable.
        (https://github.com/microsoft/DeepSpeed/pull/4966)
      - Hybrid architectures (e.g., Qwen3.5): same decoder layer class but different
        child submodules per instance (self_attn vs linear_attn).

    Marking these as z3 leaves forces whole-module allgather instead of per-submodule
    prefetch, fixing the issue at the cost of slightly higher peak memory.
    """
    z3_leaf_classes = set()
    child_sigs: dict[type, frozenset[str]] = {}

    for m in model.modules():
        # MoE: dynamic expert routing
        if "SparseMoeBlock" in m.__class__.__name__:
            z3_leaf_classes.add(m.__class__)
            continue

        # Hybrid: same class, different child submodules across instances
        cls = m.__class__
        children = frozenset(name for name, _ in m.named_children())
        if not children:
            continue
        if cls in child_sigs:
            if child_sigs[cls] != children:
                z3_leaf_classes.add(cls)
        else:
            child_sigs[cls] = children

    if z3_leaf_classes:
        deepspeed.utils.set_z3_leaf_modules(model, list(z3_leaf_classes))
        for cls in z3_leaf_classes:
            print(f"Setting zero3 leaf: {cls.__name__}")


