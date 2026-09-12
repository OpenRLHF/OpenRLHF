"""Exercise CLI loader calls without initializing training models or CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_deepspeed_save_model import ds_module  # noqa: F401


@pytest.mark.parametrize("cli", ["train_sft", "train_dpo", "train_rm"])
@pytest.mark.parametrize("size,replicas,batch_size", [(1, 1, 2), (3, 8, 1), (10, 2, 2), (8, 2, 2), (0, 2, 2)])
def test_eval_keeps_tail_and_training_still_drops_it(ds_module, monkeypatch, cli, size, replicas, batch_size):
    # Execute the actual CLI call expressions; model construction requires GPUs.
    path = Path(__file__).resolve().parents[1] / "openrlhf" / "cli" / f"{cli}.py"
    calls = {
        node.args[0].id: node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setup_dataloader"
    }
    dataset = torch.utils.data.TensorDataset(torch.arange(size))
    dataset.collate_fn = torch.utils.data.default_collate
    strategy = object.__new__(ds_module.DeepspeedStrategy)
    strategy.seed = 42
    strategy.ds_device_mesh = {"dp": SimpleNamespace(get_group=lambda: None)}
    monkeypatch.setattr(ds_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ds_module.dist, "get_world_size", lambda **_: replicas)
    args = SimpleNamespace(
        train=SimpleNamespace(micro_batch_size=batch_size), data=SimpleNamespace(dataloader_num_workers=0)
    )
    for name in ("eval_dataset", "train_dataset"):
        observed, lengths = [], []
        for rank in range(replicas):
            monkeypatch.setattr(ds_module.dist, "get_rank", lambda **_: rank)
            loader = eval(
                compile(ast.Expression(calls[name]), str(path), "eval"),
                {
                    "strategy": strategy,
                    "args": args,
                    name: dataset,
                },
            )
            lengths.append(len(loader))
            observed.extend(index for batch in loader for index in batch[0].tolist())
        assert len(set(lengths)) == 1
        if name == "eval_dataset":
            assert set(observed) == set(range(size))
        else:
            assert len(observed) == size // (replicas * batch_size) * replicas * batch_size
