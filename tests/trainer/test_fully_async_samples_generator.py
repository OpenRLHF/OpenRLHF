from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
EXPERIENCE_PATH = PROJECT_ROOT / "openrlhf/trainer/ppo_utils/experience.py"
SAMPLES_GENERATOR_PATH = PROJECT_ROOT / "openrlhf/trainer/ppo_utils/samples_generator.py"
PPO_TRAINER_ASYNC_PATH = PROJECT_ROOT / "openrlhf/trainer/ppo_trainer_async.py"


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRef:
    def __init__(self, prompt: str, label: str):
        self.responses = [{"prompt": prompt, "label": label}]


class _FakeRayController:
    def __init__(self):
        self.cancelled_refs: list[_FakeRef] = []

    def wait(self, refs, num_returns=1, timeout=None):  # noqa: ARG002
        if not refs:
            return [], []
        return [refs[0]], refs[1:]

    @staticmethod
    def get(ref):
        return ref.responses

    def cancel(self, ref):
        self.cancelled_refs.append(ref)


def _install_stub_modules() -> dict[str, types.ModuleType]:
    launcher_module = types.ModuleType("openrlhf.trainer.ray.launcher")
    launcher_module.RayActorGroup = object

    vllm_engine_module = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    vllm_engine_module.batch_vllm_engine_call = lambda *args, **kwargs: None

    models_utils_module = types.ModuleType("openrlhf.models.utils")
    models_utils_module.compute_approx_kl = lambda *args, **kwargs: None
    models_utils_module.compute_reward = lambda *args, **kwargs: None
    models_utils_module.masked_mean = lambda *args, **kwargs: None
    models_utils_module.reward_with_kl_penalty = lambda *args, **kwargs: None

    length_penalty_module = types.ModuleType("openrlhf.trainer.ppo_utils.length_penalty")
    length_penalty_module.apply_length_penalties = lambda *args, **kwargs: None

    logging_utils_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils_module.init_logger = lambda name: logging.getLogger(name)

    seqlen_balancing_module = types.ModuleType("openrlhf.utils.seqlen_balancing")
    seqlen_balancing_module.get_minimum_num_micro_batch_size = lambda *args, **kwargs: 1
    seqlen_balancing_module.get_seqlen_balanced_partitions = lambda *args, **kwargs: []

    utils_module = types.ModuleType("openrlhf.utils.utils")
    utils_module.zero_pad_sequences = lambda items, **kwargs: items

    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = _FakeSamplingParams

    return {
        "openrlhf.trainer.ray.launcher": launcher_module,
        "openrlhf.trainer.ray.vllm_engine": vllm_engine_module,
        "openrlhf.models.utils": models_utils_module,
        "openrlhf.trainer.ppo_utils.length_penalty": length_penalty_module,
        "openrlhf.utils.logging_utils": logging_utils_module,
        "openrlhf.utils.seqlen_balancing": seqlen_balancing_module,
        "openrlhf.utils.utils": utils_module,
        "vllm": vllm_module,
    }


def _load_samples_generator_module():
    stub_modules = _install_stub_modules()
    original_modules = {name: sys.modules.get(name) for name in stub_modules}
    original_experience_module = sys.modules.get("openrlhf.trainer.ppo_utils.experience")
    sys.modules.update(stub_modules)

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        experience_spec = importlib.util.spec_from_file_location(
            "openrlhf.trainer.ppo_utils.experience",
            EXPERIENCE_PATH,
        )
        experience_module = importlib.util.module_from_spec(experience_spec)
        sys.modules[experience_spec.name] = experience_module
        assert experience_spec.loader is not None
        experience_spec.loader.exec_module(experience_module)

        spec = importlib.util.spec_from_file_location("_samples_generator_under_test", SAMPLES_GENERATOR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("_samples_generator_under_test", None)
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
        if original_experience_module is None:
            sys.modules.pop("openrlhf.trainer.ppo_utils.experience", None)
        else:
            sys.modules["openrlhf.trainer.ppo_utils.experience"] = original_experience_module

    return module


def _build_samples_generator(module, prompts: list[str]):
    generator = module.SamplesGenerator.__new__(module.SamplesGenerator)
    generator.args = SimpleNamespace(
        vllm_enable_sleep=False,
        dynamic_filtering=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        enable_vllm_is_correction=False,
    )
    generator.tokenizer = None
    generator.vllm_engines = []
    generator.prompts_dataloader = [(None, [prompt], [f"label-{prompt}"], [None]) for prompt in prompts]
    generator.eval_dataloader = None
    return generator


def _wire_fake_generation(module, generator, monkeypatch: pytest.MonkeyPatch):
    dispatched_prompts = []
    dispatched_images = []
    ray_controller = _FakeRayController()

    def fake_dispatch(prompts, labels, *, images=None, **generate_kwargs):  # noqa: ARG001
        refs = []
        if images is None:
            images = [None] * len(prompts)
        for prompt, label, image in zip(prompts, labels, images):
            dispatched_prompts.append(prompt)
            dispatched_images.append(image)
            refs.append(_FakeRef(prompt, label))
        return refs

    def fake_process_response(response, **generate_kwargs):  # noqa: ARG001
        return module.Experience(prompts=[response["prompt"]], labels=[response["label"]])

    monkeypatch.setattr(module.ray, "wait", ray_controller.wait)
    monkeypatch.setattr(module.ray, "get", ray_controller.get)
    monkeypatch.setattr(module.ray, "cancel", ray_controller.cancel)
    monkeypatch.setattr(generator, "_dispatch_prompts_to_vllm", fake_dispatch)
    monkeypatch.setattr(generator, "_process_response_into_experience", fake_process_response)

    return dispatched_prompts, dispatched_images, ray_controller


@pytest.mark.unit
def test_generate_samples_fully_async_reuses_inflight_prompt_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_samples_generator_module()
    generator = _build_samples_generator(module, ["p0", "p1", "p2", "p3"])
    dispatched_prompts, _dispatched_images, ray_controller = _wire_fake_generation(module, generator, monkeypatch)

    first_batch, first_filter_pass_rate, first_prompts_consumed, first_exhausted = (
        generator.generate_samples_fully_async()
    )
    second_batch, second_filter_pass_rate, second_prompts_consumed, second_exhausted = (
        generator.generate_samples_fully_async()
    )

    assert [experience.prompts[0] for experience in first_batch] == ["p0", "p1"]
    assert first_filter_pass_rate is None
    assert first_prompts_consumed == 3
    assert not first_exhausted

    assert [experience.prompts[0] for experience in second_batch] == ["p2", "p3"]
    assert second_filter_pass_rate is None
    assert second_prompts_consumed == 1
    assert second_exhausted

    assert dispatched_prompts == ["p0", "p1", "p2", "p3"]
    assert not ray_controller.cancelled_refs


@pytest.mark.unit
def test_generate_samples_fully_async_returns_incomplete_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_samples_generator_module()
    generator = _build_samples_generator(module, ["p0", "p1", "p2"])
    dispatched_prompts, _dispatched_images, ray_controller = _wire_fake_generation(module, generator, monkeypatch)

    first_batch, _, first_prompts_consumed, first_exhausted = generator.generate_samples_fully_async()
    tail_batch, tail_filter_pass_rate, tail_prompts_consumed, tail_exhausted = generator.generate_samples_fully_async()

    assert [experience.prompts[0] for experience in first_batch] == ["p0", "p1"]
    assert first_prompts_consumed == 3
    assert not first_exhausted

    assert [experience.prompts[0] for experience in tail_batch] == ["p2"]
    assert tail_filter_pass_rate is None
    assert tail_prompts_consumed == 0
    assert tail_exhausted
    assert dispatched_prompts == ["p0", "p1", "p2"]
    assert not ray_controller.cancelled_refs


@pytest.mark.unit
def test_fully_async_path_preserves_images_when_dispatching(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_samples_generator_module()
    generator = _build_samples_generator(module, ["p0", "p1"])
    generator.prompts_dataloader = [
        (None, ["p0"], ["label-p0"], [["img-0.png"]]),
        (None, ["p1"], ["label-p1"], [["img-1.png"]]),
    ]
    dispatched_prompts, dispatched_images, _ray_controller = _wire_fake_generation(module, generator, monkeypatch)

    first_group, consumed, exhausted = generator.stream_prompt_group_fully_async()
    second_group, second_consumed, second_exhausted = generator.stream_prompt_group_fully_async()

    assert [experience.prompts[0] for experience in first_group] == ["p0"]
    assert [experience.prompts[0] for experience in second_group] == ["p1"]
    assert consumed == 2
    assert second_consumed == 0
    assert exhausted is False
    assert second_exhausted is True
    assert dispatched_prompts == ["p0", "p1"]
    assert dispatched_images == [["img-0.png"], ["img-1.png"]]


@pytest.mark.unit
def test_partial_rollout_path_uses_fully_async_generation_and_eval_pause_resume() -> None:
    module = ast.parse(PPO_TRAINER_ASYNC_PATH.read_text())
    source = PPO_TRAINER_ASYNC_PATH.read_text()

    assert "generate_samples_fully_async" in source
    assert "stream_prompt_group_fully_async" in source
    assert "pause_generation" in source
    assert "resume_generation" in source

    generate_actor = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GenerateSamplesActor"
    )
    fit_method = next(
        node for node in generate_actor.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_body"
    )
    call_names = {
        child.func.attr
        for child in ast.walk(fit_method)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }

    assert "generate_samples_fully_async" in call_names or "stream_prompt_group_fully_async" in call_names


@pytest.mark.unit
def test_process_response_into_experience_uses_partial_rollout_mask_and_versions() -> None:
    module = _load_samples_generator_module()
    generator = _build_samples_generator(module, [])

    response = {
        "prompt": "prompt",
        "label": "label",
        "observation_tokens": [10, 11, 12, 13, 14],
        "action_ranges": [(2, 5)],
        "action_loss_mask": [0, 1, 1],
        "rollout_log_probs": [0.0, 0.0, -0.1, -0.2, -0.3],
        "reward": 1.5,
        "scores": 1.0,
        "truncated": False,
        "min_weight_version": 3,
        "max_weight_version": 4,
        "partial_old_token_count": 1,
        "extra_logs": {},
    }

    experience = generator._process_response_into_experience(response, max_len=8)

    assert torch.equal(experience.action_mask, torch.tensor([[0, 0, 1, 1]], dtype=torch.bool))
    assert experience.response_length.item() == 3
    assert experience.info["min_weight_version"].item() == 3
    assert experience.info["max_weight_version"].item() == 4
    assert experience.info["partial_old_token_count"].item() == 1
