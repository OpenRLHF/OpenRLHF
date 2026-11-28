"""Check preference eval on CPU with fixed model scores."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@pytest.mark.parametrize("trainer_name", ["rm", "rm_logexp", "dpo"])
@pytest.mark.parametrize("use_margin", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("metric,expected", [("acc_mean", 2 / 3), ("eval_loss", 0.91781775)])
def test_preference_eval_weights_each_pair_equally(trainer_name, batch_size, metric, expected, use_margin):
    root = Path(__file__).resolve().parents[1]
    is_rm = trainer_name.startswith("rm")
    path = root / "openrlhf" / "trainer" / f"{'rm' if is_rm else 'dpo'}_trainer.py"
    tree = ast.parse(path.read_text())
    evaluate = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    namespace = {"torch": torch, "tqdm": tqdm, "nn": nn, "F": F}
    # Use the real evaluator and losses. Their module imports need GPU dependencies.
    loss_tree = ast.parse((root / "openrlhf" / "models" / "loss.py").read_text())
    loss_class = {"rm": "PairWiseLoss", "rm_logexp": "LogExpLoss", "dpo": "DPOLoss"}[trainer_name]
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
        margin_loss=use_margin,
        loss_fn=namespace[loss_class]() if is_rm else namespace[loss_class](beta=1.0),
        _wandb=SimpleNamespace(log=logged.append),
        _tensorboard=None,
    )

    def forward(current_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, *args):
        chosen, rejected = chosen_ids[:, 0].float(), rejected_ids[:, 0].float()
        if current_model is reference:
            chosen, rejected = torch.zeros_like(chosen), torch.zeros_like(rejected)
        return (chosen, rejected, None) if is_rm else (chosen, rejected, None, None)

    trainer.concatenated_forward = forward
    # The first two pairs are ranked correctly; the last pair is incorrect and has a larger loss.
    chosen = torch.tensor([[-1], [-1], [-4]])
    rejected = torch.tensor([[-2], [-2], [-2]])
    dataset = torch.utils.data.TensorDataset(
        chosen.unsqueeze(1),
        torch.ones_like(chosen).unsqueeze(1),
        rejected.unsqueeze(1),
        torch.ones_like(rejected).unsqueeze(1),
        torch.full((3,), 2.0 if use_margin else 0.0),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)
    namespace["evaluate"](trainer, loader, steps=7)

    if is_rm and use_margin and metric == "eval_loss":
        expected = -F.logsigmoid(torch.tensor([-1.0, -1.0, -4.0])).mean().item()
    assert logged[0][f"eval/{metric}"] == pytest.approx(expected)
    assert model.training
