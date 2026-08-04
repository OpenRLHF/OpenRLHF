import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_samples_generator_module(monkeypatch):
    ray_module = types.ModuleType("ray")
    monkeypatch.setitem(sys.modules, "ray", ray_module)

    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

    experience_module = types.ModuleType("openrlhf.trainer.ppo_utils.experience")
    experience_module.Experience = object
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", experience_module)

    engine_module = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    engine_module.batch_vllm_engine_call = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.vllm_engine", engine_module)

    logging_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", logging_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.trainer.ppo_utils.samples_generator",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_filter_pass_rate_counts_accepted_prompts(monkeypatch):
    module = _load_samples_generator_module(monkeypatch)
    args = SimpleNamespace(
        vllm=SimpleNamespace(enable_sleep=False),
        rollout=SimpleNamespace(batch_size=1, n_samples_per_prompt=8, vllm_generate_batch_size=1),
        algo=SimpleNamespace(dynamic_filtering_enable=True),
    )
    generator = module.SamplesGenerator(
        strategy=SimpleNamespace(args=args),
        prompts_dataloader=[None],
        eval_dataloader=None,
        tokenizer=None,
        vllm_engines=[],
    )
    generator._generate_vllm = lambda **kwargs: ([object()] * 8, 1, False)

    samples, pass_rate, prompts_consumed, exhausted = generator.generate_samples(n_samples_per_prompt=8)

    assert len(samples) == 8
    assert pass_rate == 100.0
    assert prompts_consumed == 1
    assert not exhausted
