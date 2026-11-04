"""Check preference eval on CPU with fixed model scores."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@pytest.mark.parametrize("trainer_name", ["rm", "dpo"])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("metric,expected", [("acc_mean", 2 / 3), ("eval_loss", 0.91781775)])
def test_preference_eval_weights_each_pair_equally(trainer_name, batch_size, metric, expected):
    root = Path(__file__).resolve().parents[1]
    path = root / "openrlhf" / "trainer" / f"{trainer_name}_trainer.py"
    tree = ast.parse(path.read_text())
    evaluate = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    namespace = {"torch": torch, "tqdm": tqdm, "nn": nn, "F": F}
    # Use the real evaluator and losses. Their module imports need GPU dependencies.
    loss_tree = ast.parse((root / "openrlhf" / "models" / "loss.py").read_text())
    loss_class = "PairWiseLoss" if trainer_name == "rm" else "DPOLoss"
    loss_node = next(node for node in loss_tree.body if isinstance(node, ast.ClassDef) and node.name == loss_class)
    namespace["Tuple"] = tuple
    exec(compile(ast.Module(body=[evaluate, loss_node], type_ignores=[]), str(path), "exec"), namespace)

    model = nn.Linear(1, 1)
    model.config = SimpleNamespace()
    reference = nn.Linear(1, 1)
    logged = []
    strategy = SimpleNamespace(
        is_rank_0=lambda: True,
        all_reduce=lambda values: values,
        all_gather=lambda values: values,
        _unwrap_model=lambda value: value,
        print=lambda *args: None,
    )
    trainer = SimpleNamespace(
        model=model,
        ref_model=reference,
        strategy=strategy,
        margin_loss=False,
        loss_fn=namespace[loss_class]() if trainer_name == "rm" else namespace[loss_class](beta=1.0),
        _wandb=SimpleNamespace(log=logged.append),
        _tensorboard=None,
    )

    def forward(current_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, *args):
        chosen, rejected = chosen_ids[:, 0].float(), rejected_ids[:, 0].float()
        if current_model is reference:
            chosen, rejected = torch.zeros_like(chosen), torch.zeros_like(rejected)
        return (chosen, rejected, None) if trainer_name == "rm" else (chosen, rejected, None, None)

    trainer.concatenated_forward = forward
    # The first two pairs are ranked correctly; the last pair is incorrect and has a larger loss.
    chosen = torch.tensor([[-1], [-1], [-4]])
    rejected = torch.tensor([[-2], [-2], [-2]])
    dataset = torch.utils.data.TensorDataset(
        chosen.unsqueeze(1),
        torch.ones_like(chosen).unsqueeze(1),
        rejected.unsqueeze(1),
        torch.ones_like(rejected).unsqueeze(1),
        torch.zeros(3),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)
    namespace["evaluate"](trainer, loader, steps=7)

    assert logged[0][f"eval/{metric}"] == pytest.approx(expected)
    assert model.training


@pytest.mark.parametrize("trainer_name", ["rm", "dpo"])
@pytest.mark.parametrize("scheduler_name", ["cosine_with_min_lr", "linear"])
@pytest.mark.parametrize(
    "size,micro_batch,dp_size,gas,epochs",
    [
        (10, 1, 1, 4, 3),
        (11, 2, 1, 2, 3),
        (23, 2, 2, 2, 3),
        (12, 2, 1, 2, 3),
        (1000, 8, 1, 16, 3),
        (1000, 8, 1, 16, 1),
        (11, 2, 1, 1, 3),
    ],
)
def test_preference_cli_scheduler_matches_updates(
    monkeypatch, tmp_path, trainer_name, scheduler_name, size, micro_batch, dp_size, gas, epochs
):
    import math
    import sys
    from unittest.mock import MagicMock

    from torchdata.stateful_dataloader import StatefulDataLoader
    from transformers import get_scheduler

    from openrlhf.utils.distributed_sampler import DistributedSampler
    from tests.test_sft_trainer import _Engine, _load_module

    dataset = torch.utils.data.TensorDataset(torch.arange(size).float())
    dataset.collate_fn = torch.utils.data.default_collate
    sampler = DistributedSampler(dataset, num_replicas=dp_size, rank=0, drop_last=True)
    loader = StatefulDataLoader(dataset, batch_size=micro_batch, sampler=sampler, drop_last=True)
    strategy = MagicMock(accumulated_gradient=gas)
    strategy.setup_dataloader.return_value = loader
    prepared = (MagicMock(), MagicMock(), MagicMock())
    strategy.prepare.return_value = (prepared, MagicMock()) if trainer_name == "dpo" else prepared
    for name in [
        "openrlhf.datasets",
        "openrlhf.datasets.utils",
        "openrlhf.models",
        f"openrlhf.trainer.{trainer_name}_trainer",
        "openrlhf.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    root = Path(__file__).resolve().parents[1]
    module = _load_module(monkeypatch, "_preference_scheduler_cli", root / f"openrlhf/cli/train_{trainer_name}.py")
    module.get_strategy.return_value = strategy
    module.RewardDataset.return_value = dataset
    args = MagicMock()
    args.model.gradient_checkpointing_enable = False
    args.eval.dataset = None
    args.ckpt.load_enable = False
    args.ckpt.output_dir = str(tmp_path)
    args.train.max_epochs = epochs
    args.train.batch_size = micro_batch * dp_size * gas
    args.data.max_samples = size
    args.lr_scheduler = scheduler_name
    args.lr_warmup_ratio = 0.2
    args.min_lr_ratio = 0.1
    module.train(args)
    config = strategy.prepare.call_args.args[0][1]
    engine = _Engine(gas)
    warmup_steps = math.ceil(config["scheduler_steps"] * config["lr_warmup_ratio"])
    engine.scheduler = get_scheduler(
        config["lr_scheduler"],
        engine.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config["scheduler_steps"],
        scheduler_specific_kwargs=(
            {"min_lr_rate": config["min_lr_ratio"]} if scheduler_name == "cosine_with_min_lr" else {}
        ),
    )
    rates = [engine.optimizer.param_groups[0]["lr"]]
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for (values,) in loader:
            engine.backward(engine.weight * (values.mean() + 1))
            engine.step()
            if engine.micro_steps % gas == 0:
                rates.append(engine.optimizer.param_groups[0]["lr"])
    assert config["scheduler_steps"] == engine.global_steps
    assert engine.scheduler.last_epoch == engine.global_steps
    assert rates[0] == 0
    assert rates[warmup_steps] == pytest.approx(0.01)
    assert all(a <= b for a, b in zip(rates[:warmup_steps], rates[1 : warmup_steps + 1]))
    assert all(a >= b for a, b in zip(rates[warmup_steps:], rates[warmup_steps + 1 :]))
    assert rates[-1] == pytest.approx(0.001 if scheduler_name == "cosine_with_min_lr" else 0)
