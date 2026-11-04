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
@pytest.mark.parametrize(
    "size,micro_batch_size,batch_size", [(10, 1, 4), (10, 2, 4), (11, 2, 4), (12, 1, 4), (12, 2, 4), (1000, 8, 128)]
)
@pytest.mark.parametrize("checkpoint_position", ["first", "cross_epoch", "near_end"])
def test_preference_resume_matches_continuous_training(
    trainer_name, size, micro_batch_size, batch_size, checkpoint_position, tmp_path
):
    import copy
    from unittest.mock import MagicMock

    from openrlhf.utils.distributed_sampler import DistributedSampler
    from tests.test_sft_trainer import _Actor, _Strategy

    root = Path(__file__).resolve().parents[1]
    path = root / "openrlhf/trainer" / f"{trainer_name}_trainer.py"
    fit = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    loss_name = "PairWiseLoss" if trainer_name == "rm" else "DPOLoss"
    loss = next(
        node
        for node in ast.parse((root / "openrlhf/models/loss.py").read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == loss_name
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tuple": tuple,
        "tqdm": MagicMock(),
        "DistributedSampler": DistributedSampler,
    }
    exec(compile(ast.Module(body=[fit, loss], type_ignores=[]), str(path), "exec"), namespace)
    gas = batch_size // micro_batch_size
    samples_per_epoch = size // micro_batch_size * micro_batch_size
    checkpoint_step = {
        "first": 1,
        "cross_epoch": samples_per_epoch // batch_size + 1,
        "near_end": samples_per_epoch * 3 // batch_size - 1,
    }[checkpoint_position]
    checkpoint_path = tmp_path / "checkpoint.pt"

    def run(resume=False):
        ids = torch.arange(size).reshape(-1, 1, 1)
        dataset = torch.utils.data.TensorDataset(
            ids, torch.ones_like(ids), ids, torch.ones_like(ids), torch.zeros(size)
        )
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42, drop_last=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler, drop_last=True)
        model = _Actor(gas)
        engine = model.model
        engine.batch_size = batch_size
        reference = _Actor(gas)
        reference.model.weight.data.zero_()
        consumed = 0
        prefix = []
        if resume:
            state = torch.load(checkpoint_path, weights_only=True)
            engine.load_state_dict(state["model"])
            engine.optimizer.load_state_dict(state["optimizer"])
            engine.scheduler.load_state_dict(state["scheduler"])
            engine.micro_steps = state["micro_steps"]
            engine.global_steps = state["global_steps"]
            engine.global_samples = state["global_samples"]
            consumed = state["client_states"]["consumed_samples"]
            prefix = state["seen"]
            assert consumed == len(prefix)
        args = SimpleNamespace(
            train=SimpleNamespace(batch_size=batch_size),
            model=SimpleNamespace(aux_loss_coef=0, nll_loss_coef=0),
            eval=SimpleNamespace(steps=-1),
            ckpt=SimpleNamespace(save_steps=-1),
        )
        trainer = SimpleNamespace(
            args=args,
            strategy=_Strategy(gas),
            train_dataloader=loader,
            epochs=3,
            model=model,
            ref_model=reference,
            optimizer=engine.optimizer,
            scheduler=engine.scheduler,
            aux_loss=False,
            nll_loss=False,
            margin_loss=False,
            compute_fp32_loss=True,
            _wandb=None,
            _tensorboard=None,
            loss_fn=namespace[loss_name]() if trainer_name == "rm" else namespace[loss_name](beta=0.1),
        )

        def forward(current_model, chosen, *args):
            if current_model is model:
                model.seen.extend(chosen[:, 0].tolist())
            scores = current_model.model.weight * (chosen[:, 0].float() + 1) / (size + 1)
            return (scores, -scores, None) if trainer_name == "rm" else (scores, -scores, None, None)

        trainer.concatenated_forward = forward

        def save(args, step, bar, logs, client_states):
            if not resume and step == checkpoint_step:
                torch.save(
                    copy.deepcopy(
                        {
                            "model": engine.state_dict(),
                            "optimizer": engine.optimizer.state_dict(),
                            "scheduler": engine.scheduler.state_dict(),
                            "micro_steps": engine.micro_steps,
                            "global_steps": engine.global_steps,
                            "global_samples": engine.global_samples,
                            "client_states": client_states,
                            "seen": model.seen,
                        }
                    ),
                    checkpoint_path,
                )

        trainer.save_logs_and_checkpoints = save
        namespace["fit"](trainer, args, consumed_samples=consumed, num_update_steps_per_epoch=size // batch_size)
        return engine, prefix + model.seen

    continuous, expected_order = run()
    resumed, actual_order = run(resume=True)
    assert actual_order == expected_order
    assert resumed.global_steps == continuous.global_steps
    assert resumed.global_samples == continuous.global_samples
    assert resumed.micro_steps == continuous.micro_steps
    assert torch.equal(resumed.weight, continuous.weight)
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    assert resumed.optimizer.param_groups[0]["lr"] == continuous.optimizer.param_groups[0]["lr"]
    assert torch.equal(
        resumed.optimizer.state[resumed.weight]["momentum_buffer"],
        continuous.optimizer.state[continuous.weight]["momentum_buffer"],
    )
