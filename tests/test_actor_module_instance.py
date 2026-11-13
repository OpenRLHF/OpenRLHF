from copy import deepcopy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .test_actor_entropy_temperature import Actor


def _model():
    torch.manual_seed(23)
    config = GPT2Config(
        vocab_size=17,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config)


def _inputs(padding):
    if padding == "left":
        sequences = torch.tensor([[0, 0, 2, 4, 6, 8], [0, 3, 5, 7, 9, 11]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    else:
        sequences = torch.tensor([[2, 4, 6, 8, 0, 0], [3, 5, 7, 9, 11, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    return sequences, attention_mask


def _reference(model, sequences, attention_mask, temperature):
    positions = attention_mask.cumsum(-1) - 1
    positions = positions.masked_fill(attention_mask == 0, 1)
    logits = model(sequences, attention_mask=attention_mask, position_ids=positions).logits.float() / temperature
    log_probs = logits.log_softmax(-1)
    next_log_probs = log_probs[:, :-1].gather(-1, sequences[:, 1:, None]).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)[:, :-1]
    return logits, next_log_probs, entropy


@pytest.mark.parametrize("packing_samples", [False, True])
def test_module_actor_retains_packing_configuration(packing_samples):
    model = _model()
    actor = Actor(model, packing_samples=packing_samples)
    assert actor.model is model
    assert actor.packing_samples is packing_samples
    assert all(actual is expected for actual, expected in zip(actor.parameters(), model.parameters()))


@pytest.mark.parametrize("mode", ["actions", "logprobs", "output"])
@pytest.mark.parametrize("padding", ["left", "right"])
@pytest.mark.parametrize("temperature", [0.7, 1.5])
def test_module_actor_forward_matches_native_model(mode, padding, temperature):
    model = _model()
    actor = Actor(model, temperature=temperature).eval()
    sequences, attention_mask = _inputs(padding)
    action_mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    with torch.no_grad():
        expected_logits, expected_log_probs, expected_entropy = _reference(
            model, sequences, attention_mask, temperature
        )
        result = actor(
            sequences,
            action_mask if mode == "actions" else None,
            attention_mask=attention_mask,
            return_output=True,
            return_logprobs=mode == "logprobs",
            return_entropy=True,
        )
    if mode == "output":
        output = result
    else:
        actual_log_probs, output = result
        expected = (
            expected_log_probs[:, -action_mask.shape[1] :] * action_mask if mode == "actions" else expected_log_probs
        )
        torch.testing.assert_close(actual_log_probs, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(output.entropy, expected_entropy, rtol=1e-5, atol=1e-6)


def _optimizer(model, optimizer_name):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.8)
    return torch.optim.AdamW(model.parameters(), lr=0.002, eps=1e-4, weight_decay=0.03)


def _assert_training_state(actual, reference, actual_optimizer, reference_optimizer):
    for parameter, expected in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=2e-6)
        actual_state = actual_optimizer.state[parameter]
        reference_state = reference_optimizer.state[expected]
        assert actual_state.keys() == reference_state.keys()
        for key in actual_state:
            torch.testing.assert_close(actual_state[key], reference_state[key], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("temperature", [1.0, 1.4])
def test_module_actor_trains_and_restores_checkpoint(tmp_path, optimizer_name, temperature):
    reference = _model()
    actor = Actor(deepcopy(reference), temperature=temperature)
    optimizer = _optimizer(actor, optimizer_name)
    reference_optimizer = _optimizer(reference, optimizer_name)
    sequences, attention_mask = _inputs("left")
    action_mask = torch.tensor([[1, 0, 1], [1, 1, 1]])
    initial = {name: value.clone() for name, value in reference.state_dict().items()}

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_log_probs = actor(sequences, action_mask, attention_mask=attention_mask)
        _, reference_log_probs, _ = _reference(reference, sequences, attention_mask, temperature)
        actual_loss = -actual_log_probs.sum() / action_mask.sum()
        expected_loss = -(reference_log_probs[:, -3:] * action_mask).sum() / action_mask.sum()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        actual_loss.backward()
        expected_loss.backward()
        for parameter, expected in zip(actor.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=2e-6)
        optimizer.step()
        reference_optimizer.step()
        _assert_training_state(actor, reference, optimizer, reference_optimizer)

        if step == 0:
            checkpoint = tmp_path / "actor.pt"
            torch.save({"model": actor.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            restored = Actor(_model(), temperature=temperature)
            restored_optimizer = _optimizer(restored, optimizer_name)
            state = torch.load(checkpoint, weights_only=True)
            restored.load_state_dict(state["model"])
            restored_optimizer.load_state_dict(state["optimizer"])
            for actual, expected in zip(restored.parameters(), actor.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actor, optimizer = restored, restored_optimizer

    assert any(not torch.equal(value, initial[name]) for name, value in reference.state_dict().items())
