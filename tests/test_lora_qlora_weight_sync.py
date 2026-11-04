"""Regression tests for LoRA/QLoRA actor-to-vLLM weight synchronization.

The tests exercise ``ActorPPOTrainer.broadcast_to_vllm`` with a recording
process group at the vLLM transport boundary. They verify the parameter name,
dtype, shape, and exact tensor value sent by the actor while avoiding a Ray
cluster or live vLLM engine.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

peft = pytest.importorskip("peft")
pytest.importorskip("deepspeed")
pytest.importorskip("ray")
pytest.importorskip("vllm")

ppo_actor = pytest.importorskip("openrlhf.trainer.ray.ppo_actor")


class _TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


class _TinyTiedActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, token_ids):
        return self.lm_head(self.proj(self.embed_tokens(token_ids)))


class _RecordingProcessGroup:
    def __init__(self, engine):
        self.engine = engine

    def broadcast(self, tensor, src, stream):
        assert src == 0
        call = self.engine.update_weight.remote.call_args
        assert call is not None
        assert tensor.dtype == call.kwargs["dtype"]
        assert tensor.shape == call.kwargs["shape"]
        self.engine.received[call.args[0]] = tensor.detach().clone()


def _make_trainer(model):
    engine = SimpleNamespace(update_weight=MagicMock(), received={})
    trainer = ppo_actor.ActorPPOTrainer.__new__(ppo_actor.ActorPPOTrainer)
    trainer.actor = SimpleNamespace(model=SimpleNamespace(module=model), is_vlm=False)
    trainer.strategy = SimpleNamespace(
        args=SimpleNamespace(
            vllm=SimpleNamespace(enable_prefix_caching=False, sync_with_ray=False),
            ds=SimpleNamespace(zero_stage=2, tensor_parallel_size=1),
        )
    )
    trainer.vllm_engines = [engine]
    trainer._model_update_group = _RecordingProcessGroup(engine)
    trainer.use_cuda_ipc = False
    return trainer, engine


@pytest.fixture
def sync_runtime(monkeypatch):
    """Replace distributed coordination while retaining the real sync logic."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(ppo_actor.deepspeed.zero, "GatheredParameters", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(ppo_actor, "torch_dist_barrier_and_cuda_sync", lambda: None)
    monkeypatch.setattr(ppo_actor.ray, "get", lambda refs: refs)


def _make_qlora_model():
    bitsandbytes = pytest.importorskip("bitsandbytes")
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.is_loaded_in_4bit = True
    model.proj = bitsandbytes.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none"),
    )
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(
            torch.arange(128, device="cuda", dtype=torch.float32).reshape(2, 64) / 1000
        )
        layer.lora_B["default"].weight.copy_(
            torch.arange(128, device="cuda", dtype=torch.float32).reshape(64, 2) / 2000
        )
    return model


def _make_multilayer_qlora_model():
    bitsandbytes = pytest.importorskip("bitsandbytes")
    torch.manual_seed(11)
    model = torch.nn.Module()
    model.is_loaded_in_4bit = True
    model.proj_in = bitsandbytes.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    model.proj_out = bitsandbytes.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    return peft.get_peft_model(
        model,
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj_in", "proj_out"], bias="none"),
    )


def test_lora_sync_sends_exact_effective_weight_and_restores_adapter(sync_runtime):
    model = peft.get_peft_model(
        _TinyActor(),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none"),
    )
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.base_layer.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
        layer.lora_A["default"].weight.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
        layer.lora_B["default"].weight.copy_(torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]))
    base_weight = layer.base_layer.weight.detach().clone()
    expected = base_weight + layer.get_delta_weight("default").detach()

    trainer, engine = _make_trainer(model)
    trainer.broadcast_to_vllm()

    assert set(engine.received) == {"proj.weight"}
    assert engine.update_weight.remote.call_args.kwargs["empty_cache"]
    torch.testing.assert_close(engine.received["proj.weight"], expected, rtol=0, atol=0)
    assert not layer.merged
    tolerance = torch.finfo(base_weight.dtype).eps
    torch.testing.assert_close(layer.base_layer.weight, base_weight, rtol=tolerance, atol=tolerance)


def test_full_parameter_sync_is_unchanged(sync_runtime):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 3, bias=False),
    )
    expected = {name: param.detach().clone() for name, param in model.named_parameters()}
    trainer, engine = _make_trainer(model)

    trainer.broadcast_to_vllm()

    assert set(engine.received) == set(expected)
    for name, weight in expected.items():
        torch.testing.assert_close(engine.received[name], weight, rtol=0, atol=0)


def test_lora_sync_sends_tied_weight_once(sync_runtime):
    model = peft.get_peft_model(
        _TinyTiedActor(),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none"),
    )
    trainer, engine = _make_trainer(model)

    trainer.broadcast_to_vllm()

    names = [call.args[0] for call in engine.update_weight.remote.call_args_list]
    assert names.count("embed_tokens.weight") + names.count("lm_head.weight") == 1
    assert set(engine.received) == {"embed_tokens.weight", "proj.weight"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 4-bit quantization requires CUDA")
def test_qlora_sync_sends_exact_materialized_bf16_weight_and_restores_adapter(sync_runtime):
    from bitsandbytes.functional import dequantize_4bit

    expected_model = _make_qlora_model()
    expected_layer = expected_model.base_model.model.proj
    expected_layer.merge(safe_merge=True)
    expected = dequantize_4bit(
        expected_layer.base_layer.weight.data,
        expected_layer.base_layer.weight.quant_state,
    ).to(torch.bfloat16)

    model = _make_qlora_model()
    layer = model.base_model.model.proj
    trainer, engine = _make_trainer(model)
    trainer.broadcast_to_vllm()

    assert set(engine.received) == {"proj.weight"}
    assert engine.update_weight.remote.call_args.kwargs["empty_cache"]
    assert engine.received["proj.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(engine.received["proj.weight"], expected, rtol=0, atol=0)
    assert not layer.merged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 4-bit quantization requires CUDA")
def test_qlora_sync_does_not_skip_temporary_merged_parameters(sync_runtime):
    model = _make_multilayer_qlora_model()
    trainer, engine = _make_trainer(model)

    trainer.broadcast_to_vllm()

    assert set(engine.received) == {"proj_in.weight", "proj_out.weight"}
    assert not model.base_model.model.proj_in.merged
    assert not model.base_model.model.proj_out.merged
