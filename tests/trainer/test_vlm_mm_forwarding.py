from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
EXPERIENCE_PATH = PROJECT_ROOT / "openrlhf/trainer/ppo_utils/experience.py"
EXPERIENCE_MAKER_PATH = PROJECT_ROOT / "openrlhf/trainer/ppo_utils/experience_maker.py"


class _FakeGroup:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def async_run_method_batch(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _load_experience_maker_module():
    launcher_module = types.ModuleType("openrlhf.trainer.ray.launcher")
    launcher_module.RayActorGroup = object

    models_utils_module = types.ModuleType("openrlhf.models.utils")
    models_utils_module.compute_approx_kl = lambda a, b, kl_estimator=None: a - b  # noqa: ARG005
    models_utils_module.compute_reward = (
        lambda reward, kl_ctl, kl, action_mask=None, reward_clip_range=None: reward
    )  # noqa: ARG005
    models_utils_module.masked_mean = lambda value, mask, dim=-1: (value * mask).sum(dim=dim) / mask.sum(
        dim=dim
    ).clamp_min(
        1
    )  # noqa: ARG005

    length_penalty_module = types.ModuleType("openrlhf.trainer.ppo_utils.length_penalty")
    length_penalty_module.apply_length_penalties = lambda *args, **kwargs: None

    logging_utils_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils_module.init_logger = lambda name: logging.getLogger(name)

    seqlen_balancing_module = types.ModuleType("openrlhf.utils.seqlen_balancing")
    seqlen_balancing_module.get_minimum_num_micro_batch_size = lambda *args, **kwargs: 1
    seqlen_balancing_module.get_seqlen_balanced_partitions = lambda *args, **kwargs: []

    utils_module = types.ModuleType("openrlhf.utils.utils")
    utils_module.zero_pad_sequences = lambda items, side="left", value=0, stack=False: (
        torch.stack(items) if stack else torch.cat(items)
    )  # noqa: ARG005

    stub_modules = {
        "openrlhf.trainer.ray.launcher": launcher_module,
        "openrlhf.models.utils": models_utils_module,
        "openrlhf.trainer.ppo_utils.length_penalty": length_penalty_module,
        "openrlhf.utils.logging_utils": logging_utils_module,
        "openrlhf.utils.seqlen_balancing": seqlen_balancing_module,
        "openrlhf.utils.utils": utils_module,
    }
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

        maker_spec = importlib.util.spec_from_file_location(
            "_experience_maker_under_test",
            EXPERIENCE_MAKER_PATH,
        )
        maker_module = importlib.util.module_from_spec(maker_spec)
        sys.modules[maker_spec.name] = maker_module
        assert maker_spec.loader is not None
        maker_spec.loader.exec_module(maker_module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("_experience_maker_under_test", None)
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
        if original_experience_module is None:
            sys.modules.pop("openrlhf.trainer.ppo_utils.experience", None)
        else:
            sys.modules["openrlhf.trainer.ppo_utils.experience"] = original_experience_module

    return maker_module, experience_module.Experience


@pytest.mark.unit
def test_vlm_mm_inputs_forwarded_to_reward_and_critic_groups() -> None:
    maker_module, Experience = _load_experience_maker_module()
    mm_inputs = {
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    sample = Experience(
        index=[0],
        sequences=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        action_mask=torch.tensor([[1, 1]], dtype=torch.bool),
        info={},
        mm_train_inputs=[mm_inputs],
    )

    actor_group = _FakeGroup([[torch.zeros(1, 2)]])
    critic_group = _FakeGroup([[torch.zeros(1, 2)]])
    reward_group = _FakeGroup([[torch.tensor([1.0])]])
    ref_group = _FakeGroup([[torch.zeros(1, 2)]])

    args = SimpleNamespace(
        advantage_estimator="gae",
        ring_attn_size=1,
        ds_tensor_parallel_size=1,
        colocate_all_models=False,
        colocate_actor_ref=False,
        colocate_critic_reward=False,
        use_dynamic_batch=False,
        micro_rollout_batch_size=1,
        use_kl_loss=False,
        kl_estimator="k1",
    )
    strategy = SimpleNamespace(args=args)
    tokenizer = SimpleNamespace(pad_token_id=0)
    kl_controller = SimpleNamespace(value=0.0)

    maker = maker_module.RemoteExperienceMaker(
        actor_group,
        critic_group,
        reward_group,
        ref_group,
        kl_controller,
        strategy,
        tokenizer,
    )

    with patch.object(maker_module.ray, "get", side_effect=lambda ref: ref), patch.object(
        maker_module.ray, "put", side_effect=lambda value: value
    ):
        experiences = maker.make_experience([sample])

    assert len(experiences) == 1
    assert actor_group.calls[0]["mm_train_inputs_list"] == [sample.mm_train_inputs]
    assert critic_group.calls[0]["mm_train_inputs_list"] == [sample.mm_train_inputs]
    assert reward_group.calls[0]["mm_train_inputs_list"] == [sample.mm_train_inputs]
    assert ref_group.calls[0]["mm_train_inputs_list"] == [sample.mm_train_inputs]
