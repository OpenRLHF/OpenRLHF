"""Dispatch/barrier ordering contract for RemoteExperienceMaker.make_experience.

The four model forwards run on Ray actor groups whose GPUs are shared according to the
``--train.colocate_*`` flags: ``colocate_all`` puts every model on one GPU set,
``colocate_actor_ref`` pairs the actor with the reference model, and
``colocate_critic_reward`` pairs the critic with the reward model.

A model must not be dispatched while a co-located model still occupies the same GPUs, so
each pair needs a barrier. A barrier that guards a *later* dispatch must not be placed in
front of an unrelated dispatch, otherwise the driver serialises work that could have
overlapped. These tests pin that ordering for every flag combination, and need neither a
GPU nor a Ray cluster nor a model checkpoint.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

STUBBED_MODULES = (
    "openrlhf",
    "openrlhf.models",
    "openrlhf.trainer",
    "openrlhf.trainer.ppo_utils",
    "openrlhf.trainer.ray",
    "openrlhf.utils",
    "openrlhf.models.utils",
    "openrlhf.trainer.ppo_utils.experience",
    "openrlhf.trainer.ppo_utils.length_penalty",
    "openrlhf.trainer.ray.launcher",
    "openrlhf.utils.logging_utils",
    "openrlhf.utils.seqlen_balancing",
)


class _RecordingGroup:
    """Stands in for RayActorGroup, recording dispatch and empty_cache calls."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def async_run_method_batch(self, method_name, **_):
        self.events.append(f"dispatch:{self.name}")
        return f"{self.name}.{method_name}"

    def async_run_method(self, method_name, **_):
        self.events.append(f"{method_name}:{self.name}")
        return f"{self.name}.{method_name}"


def _args(colocate_all=False, colocate_actor_ref=False, colocate_critic_reward=False):
    return SimpleNamespace(
        train=SimpleNamespace(
            colocate_all=colocate_all,
            colocate_actor_ref=colocate_actor_ref,
            colocate_critic_reward=colocate_critic_reward,
        ),
        ds=SimpleNamespace(ring_attn_size=1, tensor_parallel_size=1),
        algo=SimpleNamespace(
            kl=SimpleNamespace(use_loss=False, estimator="k3"),
            advantage=SimpleNamespace(estimator="group_norm"),
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=1, micro_batch_size=1),
        reward=SimpleNamespace(clip_range=0.0),
    )


@pytest.fixture
def record_events(monkeypatch):
    """Load experience_maker.py with a recording stand-in for ``ray``.

    Returns a callable that runs one make_experience pass and yields the recorded event
    sequence. ``monkeypatch`` restores ``sys.modules`` afterwards, so the stand-ins
    cannot leak into the rest of the test session.
    """
    events = []

    fake_ray = MagicMock()
    fake_ray.get.side_effect = lambda ref: events.append(f"barrier:{ref}")
    fake_ray.put.return_value = "dummy"
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    # experience_maker pulls in torch/DeepSpeed-heavy packages, so stub every one of them.
    for name in STUBBED_MODULES:
        stub = MagicMock()
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)

    tensor_fn = MagicMock(return_value=torch.zeros(1))
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.models.utils",
        MagicMock(compute_approx_kl=tensor_fn, compute_reward=tensor_fn, masked_mean=tensor_fn),
    )
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", MagicMock(Experience=object))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.launcher", MagicMock(RayActorGroup=object))
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", MagicMock(init_logger=lambda _: MagicMock()))

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_experience_maker_dispatch_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def run(
        colocate_all=False,
        colocate_actor_ref=False,
        colocate_critic_reward=False,
        with_critic=True,
        with_reference=True,
    ):
        events.clear()
        maker = module.RemoteExperienceMaker(
            actor_model_group=_RecordingGroup("actor", events),
            critic_model_group=_RecordingGroup("critic", events) if with_critic else None,
            reward_model_group=_RecordingGroup("reward", events),
            initial_model_group=_RecordingGroup("reference", events) if with_reference else None,
            kl_controller=SimpleNamespace(value=0.0),
            strategy=SimpleNamespace(args=_args(colocate_all, colocate_actor_ref, colocate_critic_reward)),
            tokenizer=SimpleNamespace(pad_token_id=0),
        )
        maker._flatten_results = lambda refs, duplicate_factor: [torch.zeros(1)]
        maker.make_experience(
            [
                SimpleNamespace(
                    rewards=None,
                    action_mask=torch.zeros(1),
                    sequences=torch.zeros(1),
                    attention_mask=torch.zeros(1),
                    mm_train_inputs=None,
                    info={},
                )
            ]
        )
        return list(events)

    return run


def test_dispatches_every_model_forward_once(record_events):
    events = record_events(colocate_actor_ref=True)
    assert [e for e in events if e.startswith("dispatch:")] == [
        "dispatch:reward",
        "dispatch:actor",
        "dispatch:critic",
        "dispatch:reference",
    ]


def test_colocate_actor_ref_does_not_delay_the_critic(record_events):
    """The critic shares no GPUs with the actor, so it must be dispatched before that barrier."""
    events = record_events(colocate_actor_ref=True)

    assert events.index("dispatch:critic") < events.index("barrier:actor.forward")
    assert events.index("barrier:actor.forward") < events.index("dispatch:reference")


def test_colocate_critic_reward_does_not_delay_the_reference(record_events):
    """The reference shares no GPUs with the critic, so it must be dispatched before that barrier."""
    events = record_events(colocate_critic_reward=True)

    assert events.index("dispatch:reference") < events.index("barrier:critic.forward")
    assert events.index("barrier:reward.forward") < events.index("dispatch:critic")


def test_both_flags_overlap_critic_with_actor_and_reference_with_critic(record_events):
    events = record_events(colocate_actor_ref=True, colocate_critic_reward=True)

    assert events.index("dispatch:critic") < events.index("barrier:actor.forward")
    assert events.index("dispatch:reference") < events.index("barrier:critic.forward")


def test_colocate_all_is_sequential_and_ignores_the_individual_flags(record_events):
    """On one shared GPU set every forward must be released before the next dispatch."""
    events = record_events(colocate_all=True)

    for earlier, later in (("reward", "actor"), ("actor", "critic"), ("critic", "reference")):
        assert events.index(f"empty_cache:{earlier}") < events.index(f"dispatch:{later}")

    with_flags = record_events(colocate_all=True, colocate_actor_ref=True, colocate_critic_reward=True)
    assert with_flags == events


@pytest.mark.parametrize("colocate_actor_ref", [False, True])
def test_reference_cache_is_released_only_when_it_shares_gpus(record_events, colocate_actor_ref):
    events = record_events(colocate_actor_ref=colocate_actor_ref)

    if colocate_actor_ref:
        assert events.index("dispatch:reference") < events.index("empty_cache:reference")
    else:
        assert "empty_cache:reference" not in events


@pytest.mark.parametrize("with_critic", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
def test_missing_model_groups_still_dispatch_the_remaining_forwards(record_events, with_critic, with_reference):
    events = record_events(
        colocate_actor_ref=True,
        colocate_critic_reward=True,
        with_critic=with_critic,
        with_reference=with_reference,
    )

    expected = ["dispatch:reward", "dispatch:actor"]
    if with_critic:
        expected.append("dispatch:critic")
    if with_reference:
        expected.append("dispatch:reference")

    assert [e for e in events if e.startswith("dispatch:")] == expected
    assert "barrier:dummy" not in events
