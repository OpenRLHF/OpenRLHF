import argparse
import sys
from unittest.mock import MagicMock

from openrlhf.utils.config import hierarchize
from openrlhf.utils.logging_utils import WandbLogger


def test_wandb_config_is_nested_dict(monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor.optim.lr", type=float, default=1e-6)
    parser.add_argument("--logger.wandb.project", default="openrlhf")
    parser.add_argument("--seed", type=int, default=42)
    for name in ("org", "group", "run_name"):
        parser.add_argument(f"--logger.wandb.{name}", default=None)
    args = hierarchize(parser.parse_args([]))

    wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    WandbLogger(args)

    # W&B stores non-dict values via str(), so a nested namespace would become one opaque string.
    assert wandb.init.call_args.kwargs["config"] == {
        "actor": {"optim": {"lr": 1e-6}},
        "logger": {"wandb": {"project": "openrlhf", "org": None, "group": None, "run_name": None}},
        "seed": 42,
    }
