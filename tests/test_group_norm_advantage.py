"""Tests for group_norm advantage computation in RemoteExperienceMaker.

A prompt group whose rewards are all equal carries no learning signal, so
its advantages must be exactly zero. In float32 the group mean/std of
identical values are not exact, and dividing by (std + 1e-9) used to
amplify that residue to advantages close to +-1.
"""

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _models_utils_stub():
    """Build openrlhf.models.utils from the real function sources.

    The real module imports deepspeed, which the advantage computation
    does not need; exec only the pure helpers it uses.
    """
    source = (ROOT / "openrlhf" / "models" / "utils.py").read_text()
    tree = ast.parse(source)
    wanted = {"compute_reward", "masked_mean", "compute_approx_kl"}
    module = types.ModuleType("openrlhf.models.utils")
    namespace = {"torch": torch, "F": F, "Optional": Optional, "Tuple": Tuple, "Union": Union}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, "openrlhf/models/utils.py", "exec"), namespace)
            setattr(module, node.name, namespace[node.name])
    return module


def _load_experience_maker():
    def zero_pad_sequences(sequences, side="left", value=0, stack=False):
        max_len = max(sequence.size(-1) for sequence in sequences)
        padded = []
        for sequence in sequences:
            pad_len = max_len - sequence.size(-1)
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            padded.append(F.pad(sequence, padding, value=value))
        return torch.stack(padded) if stack else torch.cat(padded)

    stubs = {
        "ray": types.ModuleType("ray"),
        "openrlhf": types.ModuleType("openrlhf"),
        "openrlhf.models": types.ModuleType("openrlhf.models"),
        "openrlhf.models.utils": _models_utils_stub(),
        "openrlhf.trainer": types.ModuleType("openrlhf.trainer"),
        "openrlhf.trainer.ppo_utils": types.ModuleType("openrlhf.trainer.ppo_utils"),
        "openrlhf.trainer.ray": types.ModuleType("openrlhf.trainer.ray"),
        "openrlhf.utils": types.ModuleType("openrlhf.utils"),
        "openrlhf.utils.utils": types.ModuleType("openrlhf.utils.utils"),
        "openrlhf.utils.logging_utils": types.ModuleType("openrlhf.utils.logging_utils"),
        "openrlhf.utils.seqlen_balancing": types.ModuleType("openrlhf.utils.seqlen_balancing"),
    }
    stubs["openrlhf.utils.utils"].zero_pad_sequences = zero_pad_sequences
    stubs["openrlhf.utils.logging_utils"].init_logger = logging.getLogger
    launcher = types.ModuleType("openrlhf.trainer.ray.launcher")
    launcher.RayActorGroup = type("RayActorGroup", (), {})
    stubs["openrlhf.trainer.ray.launcher"] = launcher
    balancing = stubs["openrlhf.utils.seqlen_balancing"]
    balancing.get_minimum_num_micro_batch_size = lambda *a, **k: None
    balancing.get_seqlen_balanced_partitions = lambda *a, **k: None

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        _load_module(
            "openrlhf.trainer.ppo_utils.experience",
            ROOT / "openrlhf" / "trainer" / "ppo_utils" / "experience.py",
        )
        _load_module(
            "openrlhf.trainer.ppo_utils.length_penalty",
            ROOT / "openrlhf" / "trainer" / "ppo_utils" / "length_penalty.py",
        )
        return _load_module(
            "openrlhf.trainer.ppo_utils.experience_maker",
            ROOT / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py",
        )
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_maker_module = _load_experience_maker()
RemoteExperienceMaker = _maker_module.RemoteExperienceMaker
Experience = sys.modules["openrlhf.trainer.ppo_utils.experience"].Experience


def _make_args(n_samples_per_prompt):
    return SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=n_samples_per_prompt),
        algo=SimpleNamespace(advantage=SimpleNamespace(estimator="group_norm", gamma=1.0)),
        reward=SimpleNamespace(overlong_buffer_len=None, stop_properly_penalty_coef=None, clip_range=None),
    )


def _run_group_norm(reward_groups):
    """Run compute_advantages_and_returns for one Experience holding all groups."""
    n = len(reward_groups[0])
    assert all(len(group) == n for group in reward_groups)
    rewards = torch.tensor([r for group in reward_groups for r in group], dtype=torch.float32)
    batch = rewards.numel()
    action_len = 4
    experience = Experience(
        sequences=torch.zeros((batch, action_len + 1), dtype=torch.long),
        attention_mask=torch.ones((batch, action_len + 1), dtype=torch.long),
        action_mask=torch.ones((batch, action_len), dtype=torch.bool),
        kl=torch.zeros((batch, action_len)),
        rewards=rewards,
        index=list(range(batch)),
    )
    maker = object.__new__(RemoteExperienceMaker)
    maker.strategy = SimpleNamespace(args=_make_args(n))
    maker.kl_ctl = SimpleNamespace(value=0.0)
    maker.advantage_estimator = "group_norm"
    (result,) = maker.compute_advantages_and_returns([experience])
    return result.advantages.reshape(len(reward_groups), n, action_len)


@pytest.mark.parametrize("reward", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("n", [8, 16, 64])
def test_group_norm_uniform_group_has_exactly_zero_advantage(reward, n):
    advantages = _run_group_norm([[reward] * n])
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_group_norm_uniform_group_next_to_mixed_group():
    mixed = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    advantages = _run_group_norm([[0.3] * 8, mixed])
    assert torch.equal(advantages[0], torch.zeros_like(advantages[0]))

    group = torch.tensor(mixed)
    expected = (group - group.mean()) / (group.std() + 1e-9)
    assert torch.allclose(advantages[1, :, 0], expected, atol=1e-6)
    assert advantages[1, :, 0].abs().max() > 1.0  # the mixed group keeps its signal
