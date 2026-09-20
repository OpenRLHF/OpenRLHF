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


@pytest.mark.parametrize("ipo,label_smoothing", [(False, 0.0), (False, 0.1), (True, 0.0)])
@pytest.mark.parametrize("response_lengths", [([1, 1], [1, 1]), ([2, 2], [2, 2]), ([1, 3], [2, 4])])
@pytest.mark.parametrize("extra_padding", [0, 3])
def test_dpo_forward_normalizes_only_ipo(monkeypatch, ipo, label_smoothing, response_lengths, extra_padding):
    import sys

    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    from tests.test_loss_aggregation import _loss_module
    from tests.test_sft_trainer import _load_module

    root = Path(__file__).resolve().parents[1]
    # Isolate optional GPU package imports; load the full native trainer and sampler.
    monkeypatch.setitem(sys.modules, "openrlhf.models", SimpleNamespace(DPOLoss=_loss_module.DPOLoss))
    _load_module(monkeypatch, "openrlhf.utils.distributed_sampler", root / "openrlhf/utils/distributed_sampler.py")
    module = _load_module(monkeypatch, "_dpo_normalization_test", root / "openrlhf/trainer/dpo_trainer.py")
    trainer = module.DPOTrainer.__new__(module.DPOTrainer)
    trainer.args = SimpleNamespace(model=SimpleNamespace(ipo_enable=ipo))
    trainer.strategy = SimpleNamespace(ring_attn_group=None)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.loss_fn = _loss_module.DPOLoss(beta=0.2, label_smoothing=label_smoothing, ipo=ipo)
    prompt_lengths = [1, 2]
    chosen_lengths, rejected_lengths = response_lengths
    groups = []
    for offset, lengths in zip([0, 5], response_lengths):
        rows = [
            torch.cat([torch.arange(1, prompt + 1), torch.arange(prompt + 1, prompt + length + 1) + offset])
            for prompt, length in zip(prompt_lengths, lengths)
        ]
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        groups.append(F.pad(ids, (0, extra_padding)))
    chosen, rejected = groups
    weight = torch.tensor(0.1, requires_grad=True)
    aux = torch.tensor(0.7)

    def policy(ids, **kwargs):
        return -4 + weight * ids[:, 1:].float(), MoeCausalLMOutputWithPast(aux_loss=aux)

    def reference(ids, **kwargs):
        return -4 + 0.05 * ids[:, 1:].float(), {}

    pc, pr, returned_aux, nll = trainer.concatenated_forward(
        policy, chosen, chosen.ne(0), rejected, rejected.ne(0), prompt_lengths
    )
    rc, rr, _, _ = trainer.concatenated_forward(
        reference, chosen, chosen.ne(0), rejected, rejected.ne(0), prompt_lengths
    )
    # Independent per-response slices exclude prompts and padding, without reusing the trainer's masks.
    expected_policy, expected_reference, chosen_nll = [], [], []
    for group, lengths in zip(groups, [chosen_lengths, rejected_lengths]):
        for ids, prompt, length in zip(group, prompt_lengths, lengths):
            tokens = ids[prompt : prompt + length].float()
            policy_tokens = -4 + weight * tokens
            reference_tokens = -4 + 0.05 * tokens
            expected_policy.append(policy_tokens.mean() if ipo else policy_tokens.sum())
            expected_reference.append(reference_tokens.mean() if ipo else reference_tokens.sum())
            if group is chosen:
                chosen_nll.append(-policy_tokens.mean())
    ep, er = torch.stack(expected_policy), torch.stack(expected_reference)
    torch.testing.assert_close(torch.cat([pc, pr]), ep)
    torch.testing.assert_close(torch.cat([rc, rr]), er)
    torch.testing.assert_close(nll, torch.stack(chosen_nll).mean())
    assert returned_aux is aux

    gap = (ep[:2] - ep[2:]) - (er[:2] - er[2:])
    expected_loss = (
        (gap - 1 / (2 * 0.2)).square().mean()
        if ipo
        else (-(1 - label_smoothing) * F.logsigmoid(0.2 * gap) - label_smoothing * F.logsigmoid(-0.2 * gap)).mean()
    )
    actual_loss = trainer.loss_fn(pc, pr, rc, rr)[0]
    torch.testing.assert_close(actual_loss, expected_loss)
    # Include the optional chosen-NLL term to guard its independent normalization and gradient.
    actual_grad = torch.autograd.grad(actual_loss + 0.3 * nll, weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_loss + 0.3 * torch.stack(chosen_nll).mean(), weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
