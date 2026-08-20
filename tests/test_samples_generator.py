import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _Pbar:
    def __init__(self, *_args, **_kwargs):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


@pytest.fixture
def samples_generator_module(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_ray.cancelled = []
    fake_ray.wait = lambda refs, **_: (refs[:1], refs[1:])
    fake_ray.get = lambda ref: ref
    fake_ray.cancel = fake_ray.cancelled.append
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=_Pbar))

    experience_module = types.ModuleType("openrlhf.trainer.ppo_utils.experience")
    experience_module.Experience = object
    monkeypatch.setitem(sys.modules, experience_module.__name__, experience_module)

    engine_module = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    engine_module.batch_vllm_engine_call = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)

    logging_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_module.init_logger = lambda *_args, **_kwargs: SimpleNamespace(info=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, logging_module.__name__, logging_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_samples_generator_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_ray


def test_dynamic_filtering_preserves_terminal_accepted_rollouts(samples_generator_module):
    module, fake_ray = samples_generator_module
    generator = module.SamplesGenerator.__new__(module.SamplesGenerator)
    generator.args = SimpleNamespace(algo=SimpleNamespace(dynamic_filtering_range=(0.0, 1.0)))

    def group(prefix, scores):
        return [
            SimpleNamespace(name=f"{prefix}-{index}", scores=torch.tensor([score]))
            for index, score in enumerate(scores)
        ]

    refs = [group("accepted", [0.0, 1.0]), group("rejected", [0.0, 0.0]), group("pending", [0.0, 1.0])]
    generator._dispatch_prompts_to_vllm = lambda *_args, **_kwargs: refs
    generator._process_response_into_experience = lambda response, **_kwargs: response

    dataloader = iter((None, [f"prompt-{index}"], [None], [None]) for index in range(3))
    experiences, prompts_consumed, exhausted = generator._generate_vllm(
        dataloader, num_prompts=3, dynamic_filtering=True
    )

    assert [experience.name for experience in experiences] == [
        "accepted-0",
        "accepted-1",
        "pending-0",
        "pending-1",
    ]
    assert prompts_consumed == 3
    assert exhausted is True
    assert fake_ray.cancelled == []
