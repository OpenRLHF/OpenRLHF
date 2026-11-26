"""What the actor hands to vLLM's weight transfer on a sync."""

import types

import pytest
import torch

for _module in ("deepspeed", "flash_attn", "ray", "vllm"):
    pytest.importorskip(_module)

from openrlhf.trainer.ray.ppo_actor import ActorWeights


def _source(model, is_vlm=False):
    actor = types.SimpleNamespace(model=types.SimpleNamespace(module=model), is_vlm=is_vlm)
    return ActorWeights(actor, types.SimpleNamespace(tensor_parallel_size=1, zero_stage=2))


def _model():
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))


def test_sends_every_parameter_as_declared():
    model = _model()
    source = _source(model)
    sent = list(source.iter_full())

    assert [name for name, _ in sent] == [name for name, _ in model.named_parameters()]
    # vLLM sizes its receive buffers from the metadata, so it must describe exactly what is sent.
    assert [(m.name, m.dtype, m.shape) for m in source.metadata] == [(n, w.dtype, tuple(w.shape)) for n, w in sent]
    for (_, weight), param in zip(sent, model.parameters()):
        assert torch.equal(weight, param)
        assert weight.data_ptr() != param.data_ptr()  # a copy, so vLLM never holds the live parameter


def test_vlm_sends_trainable_parameters_only():
    model = _model()
    model[0].requires_grad_(False)

    assert [name for name, _ in _source(model, is_vlm=True).iter_full()] == ["1.weight", "1.bias"]
    # A text model sends frozen parameters too.
    assert len(list(_source(model).iter_full())) == 4
