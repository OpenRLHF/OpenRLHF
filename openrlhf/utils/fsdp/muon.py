from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class _AutoModelOptimizerConfig:
    _target_: Any
    _values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


def build_automodel_muon_optimizer(
    model: nn.Module,
    muon_cfg: dict[str, Any],
    adam_cfg: dict[str, Any],
    distributed_mesh=None,
):
    """Build AutoModel's Dion-family Muon optimizer from OpenRLHF CLI args."""
    from nemo_automodel.components.optim import utils as automodel_optim

    target = getattr(automodel_optim, "Muon", None)
    if target is None:
        raise RuntimeError(
            "AutoModel/Dion Muon is unavailable because the optional `dion` package is not installed. "
            "Install `dion` for --optim muon, or use --optim adam."
        ) from getattr(automodel_optim, "_import_error", None)

    muon_weight_decay = muon_cfg.get("weight_decay")
    if muon_weight_decay is None:
        muon_weight_decay = adam_cfg["weight_decay"]

    values = dict(
        lr=muon_cfg["lr"],
        mu=muon_cfg["momentum"],
        betas=tuple(adam_cfg["betas"]),
        weight_decay=muon_weight_decay,
        epsilon=adam_cfg["eps"],
        nesterov=muon_cfg["nesterov"],
        scalar_opt="adamw",
        scalar_lr=adam_cfg["lr"],
        scalar_betas=tuple(adam_cfg["betas"]),
        scalar_eps=adam_cfg["eps"],
    )

    # AutoModel filters kwargs by target signature. Keep this guard so a CLI
    # option cannot silently become a no-op when the installed Dion lacks it.
    ns_steps = muon_cfg.get("ns_steps", 5)
    if "ns_steps" in inspect.signature(target).parameters:
        values["ns_steps"] = ns_steps
    elif ns_steps != 5:
        raise ValueError("AutoModel/Dion Muon does not expose --muon.ns_steps; keep the default value 5.")

    cfg = _AutoModelOptimizerConfig(target, values)
    return automodel_optim.build_dion_optimizer(cfg, model, distributed_mesh=distributed_mesh)
