"""Async PPO checkpoint symmetry tests.

These tests verify that the async PPOTrainer persists and restores the same
checkpoint state as the sync PPOTrainer — in particular the oversampling
buffer inside SamplesGenerator.  The sync path was fixed in PR #1375; this
file locks in the async equivalent.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


# ── Shared test fixtures ────────────────────────────────────────────────────


class _TqdmPbar:
    """Minimal tqdm stand-in that supports iteration + update() + close()."""

    def __init__(self, iterable, **_kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def update(self, _n=1):
        pass

    def close(self):
        pass


def _load_async_module(monkeypatch):
    """Import ppo_trainer_async.py with Ray/vLLM/DeepSpeed mocked out."""

    fake_ray = MagicMock()
    fake_ray.Queue = MagicMock
    fake_ray.get.side_effect = lambda value: value
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    monkeypatch.setitem(
        sys.modules,
        "ray.util.queue",
        SimpleNamespace(Queue=MagicMock),
    )
    monkeypatch.setitem(
        sys.modules,
        "tqdm",
        SimpleNamespace(tqdm=_TqdmPbar),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.trainer.ppo_utils.samples_generator",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.trainer.ppo_utils.experience",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.trainer.ray.launcher",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.trainer.ray.vllm_engine",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.deepspeed",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.logging_utils",
        SimpleNamespace(init_logger=lambda *a, **kw: MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.utils",
        SimpleNamespace(get_tokenizer=lambda *a, **kw: None),
    )

    # Build a fake BasePPOTrainer that the async actors can inherit from.
    import types as _types

    fake_ppo_trainer = _types.ModuleType("openrlhf.trainer.ppo_trainer")

    class BasePPOTrainer:
        pass

    fake_ppo_trainer.BasePPOTrainer = BasePPOTrainer
    fake_ppo_trainer.compute_eval_metrics = MagicMock(return_value={})
    fake_ppo_trainer.prepare_datasets = MagicMock(return_value=(MagicMock(), None, 100))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", fake_ppo_trainer)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_ppo_trainer_async_test",
        root / "openrlhf" / "trainer" / "ppo_trainer_async.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def async_module(monkeypatch):
    return _load_async_module(monkeypatch)


@pytest.fixture
def generator_state():
    """An example generator_state dict as produced by SamplesGenerator.state_dict."""
    return {
        "sample_buffer": ["exp-a", "exp-b", "exp-c"],
        "dataloader_exhausted": False,
    }


# ── GenerateSamplesActor state_dict on save ──────────────────────────────────


def test_generate_samples_actor_client_states_includes_generator_state(async_module, generator_state):
    """GenerateSamplesActor must include generator_state in client_states it
    pushes to the rollout queue — matching the sync path."""
    GenActor = async_module.GenerateSamplesActor

    # Build a minimal GenerateSamplesActor instance.
    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1, partial_rollout_enable=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    actor._partial_rollout = False
    actor._last_eval_step = -1
    actor._eval_just_done = False
    actor.prompts_dataloader = MagicMock()
    actor.prompts_dataloader.state_dict.return_value = {"cursor": 42}
    actor.eval_dataloader = None

    # SamplesGenerator that produces one non-empty rollout and reports state.
    mock_gen = MagicMock()
    mock_gen.generate_samples.return_value = (
        ["rollout-1", "rollout-2"],  # rollout_samples (truthy => produced)
        0.5,  # filter_pass_rate
        2,  # prompts_consumed
        True,  # is_exhausted → breaks the while loop after first iteration
    )
    mock_gen.state_dict.return_value = generator_state
    actor.samples_generator = mock_gen

    # Fake Ray queue/slots/lock
    actor.rollout_queue = MagicMock()
    actor.rollout_slots = MagicMock()
    actor.rollout_slots.get.return_value = 0
    actor.vllm_lock = MagicMock()
    actor.vllm_lock.acquire.remote.return_value = None
    actor.vllm_lock.release.remote.return_value = None
    actor.generate_kwargs = {}

    actor.fit(episode=0, total_consumed_prompts=0)

    # rollout_queue.put was called with ("done", ...) at the end; before that
    # it was called with the rollout payload.  Verify the payload contained
    # the expected client_states keys including generator_state.
    payload_call = None
    for put_call in actor.rollout_queue.put.call_args_list:
        args, _kwargs = put_call
        if args and isinstance(args[0], tuple) and len(args[0]) == 4:
            payload_call = put_call
            break
    assert payload_call is not None, "rollout_queue.put should have been called with a rollout payload"

    payload_tuple = payload_call.args[0]
    rollout_samples, client_states, filter_pass_rate, generation_time = payload_tuple
    assert "generator_state" in client_states, "generator_state missing from async client_states"
    assert client_states["generator_state"] == generator_state
    assert client_states["data_loader_state_dict"] == {"cursor": 42}
    assert "episode" in client_states
    assert "total_consumed_prompts" in client_states


def test_generate_samples_actor_client_states_uses_none_when_state_dict_missing(async_module):
    """Backward compatibility: actors whose samples_generator lacks
    state_dict must still produce a valid client_states (generator_state=None)."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1, partial_rollout_enable=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    actor._partial_rollout = False
    actor._last_eval_step = -1
    actor._eval_just_done = False
    actor.prompts_dataloader = MagicMock()
    actor.prompts_dataloader.state_dict.return_value = {"cursor": 0}
    actor.eval_dataloader = None

    # Object that has generate_samples but no state_dict attribute at all.
    class _MinimalGenerator:
        def generate_samples(self, **_kwargs):
            return (["rollout-1"], 0.5, 1, True)

    actor.samples_generator = _MinimalGenerator()

    actor.rollout_queue = MagicMock()
    actor.rollout_slots = MagicMock()
    actor.rollout_slots.get.return_value = 0
    actor.vllm_lock = MagicMock()
    actor.vllm_lock.acquire.remote.return_value = None
    actor.vllm_lock.release.remote.return_value = None
    actor.generate_kwargs = {}

    actor.fit(episode=0, total_consumed_prompts=0)

    payload_call = None
    for put_call in actor.rollout_queue.put.call_args_list:
        args, _ = put_call
        if args and isinstance(args[0], tuple) and len(args[0]) == 4:
            payload_call = put_call
            break
    payload_tuple = payload_call.args[0]
    client_states = payload_tuple[1]
    assert client_states.get("generator_state") is None


# ── GenerateSamplesActor.load_state_dict recovery ───────────────────────────


def test_load_state_dict_restores_generator_state(async_module, generator_state):
    """load_state_dict with generator_state calls samples_generator.load_state_dict."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(partial_rollout_enable=False))
    actor.prompts_dataloader = MagicMock()
    actor.samples_generator = MagicMock()

    actor.load_state_dict({"cursor": 42}, generator_state=generator_state)

    actor.prompts_dataloader.load_state_dict.assert_called_once_with({"cursor": 42})
    actor.samples_generator.load_state_dict.assert_called_once_with(generator_state)


def test_load_state_dict_without_generator_state_still_works(async_module):
    """Old checkpoint that omitted generator_state must not crash."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(partial_rollout_enable=False))
    actor.prompts_dataloader = MagicMock()
    actor.samples_generator = MagicMock()

    # Call with just the dataloader state (old signature / old checkpoint).
    actor.load_state_dict({"cursor": 10})

    actor.prompts_dataloader.load_state_dict.assert_called_once_with({"cursor": 10})
    actor.samples_generator.load_state_dict.assert_not_called()


def test_load_state_dict_with_none_generator_state_skips_restore(async_module):
    """generator_state=None must NOT trigger a load_state_dict call."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(partial_rollout_enable=False))
    actor.prompts_dataloader = MagicMock()
    actor.samples_generator = MagicMock()

    actor.load_state_dict({"cursor": 10}, generator_state=None)

    actor.prompts_dataloader.load_state_dict.assert_called_once()
    actor.samples_generator.load_state_dict.assert_not_called()


# ── PPOTrainerAsync.fit() recovery wiring ────────────────────────────────────


def _build_ppotrainer_async(async_module, checkpoint_states, queue_size=1):
    """Construct a PPOTrainerAsync that will call fit() with the given
    checkpoint_states from init_checkpoint_states."""
    TrainerAsync = async_module.PPOTrainerAsync

    trainer = TrainerAsync.__new__(TrainerAsync)
    trainer.args = SimpleNamespace(train=SimpleNamespace(async_queue_size=queue_size))

    # Fake rollout_queue and rollout_slots so fit() can proceed.
    trainer.rollout_queue = MagicMock()
    trainer.rollout_slots = MagicMock()
    trainer.rollout_slots.get.side_effect = [0, "BLOCK"]  # second call blocks → easy exit path
    # Put a dummy rollout immediately so the loop processes one item.
    trainer.rollout_queue.get.side_effect = [
        MagicMock(),  # TrainingActor's first queue.get returns something
    ]

    # Fake actors
    trainer.generator_actor = MagicMock()
    trainer.trainer_actor = MagicMock()

    # init_checkpoint_states returns our desired state
    trainer.trainer_actor.init_checkpoint_states.remote.return_value = checkpoint_states
    trainer.trainer_actor.restore_best_checkpoint_state.remote.return_value = None
    trainer.trainer_actor.broadcast_to_vllm.remote.return_value = None
    trainer.trainer_actor.fit.remote.return_value = None
    # TrainingActor.fit accepts global_step as keyword argument.
    trainer.trainer_actor.fit.remote.side_effect = lambda global_step=0: None

    trainer.generator_actor.fit.remote.return_value = None
    trainer.generator_actor.load_state_dict.remote.return_value = None
    trainer.generator_actor.get_max_steps.remote.return_value = 100

    return trainer


def test_async_fit_restores_generator_state_from_checkpoint(async_module, generator_state):
    """PPOTrainerAsync.fit() must pass generator_state to generator_actor
    on resume — mirroring sync PPOTrainer."""
    checkpoint_states = {
        "episode": 2,
        "global_step": 10,  # non-zero → triggers the resume block
        "total_consumed_prompts": 42,
        "data_loader_state_dict": {"cursor": 7},
        "generator_state": generator_state,
    }

    trainer = _build_ppotrainer_async(async_module, checkpoint_states)
    trainer.fit()

    # Verify load_state_dict.remote was called with generator_state kwarg.
    call_args = trainer.generator_actor.load_state_dict.remote.call_args
    assert call_args is not None, "generator_actor.load_state_dict.remote was never called"
    pos_args, kw_args = call_args
    assert pos_args == ({"cursor": 7},)
    assert kw_args.get("generator_state") == generator_state


def test_async_fit_skips_restore_when_global_step_is_zero(async_module, generator_state):
    """Cold start (global_step=0) must NOT call generator_actor.load_state_dict."""
    checkpoint_states = {
        "episode": 0,
        "global_step": 0,
        "total_consumed_prompts": 0,
        "data_loader_state_dict": {},
        "generator_state": generator_state,
    }

    trainer = _build_ppotrainer_async(async_module, checkpoint_states)
    trainer.fit()

    trainer.generator_actor.load_state_dict.remote.assert_not_called()


def test_async_fit_handles_old_checkpoint_without_generator_state(async_module):
    """Older checkpoints that lack the generator_state key must resume gracefully
    (load_state_dict is still called, but generator_state defaults to None)."""
    checkpoint_states = {
        "episode": 1,
        "global_step": 5,
        "total_consumed_prompts": 20,
        "data_loader_state_dict": {"cursor": 3},
        # NOTE: no "generator_state" key — old checkpoint shape
    }

    trainer = _build_ppotrainer_async(async_module, checkpoint_states)
    trainer.fit()

    call_args = trainer.generator_actor.load_state_dict.remote.call_args
    pos_args, kw_args = call_args
    assert pos_args == ({"cursor": 3},)
    # .get() on a missing key returns None → generator_state=None
    assert kw_args.get("generator_state") is None


# ── End-to-end save→load roundtrip ───────────────────────────────────────────


def test_async_save_then_load_roundtrip_preserves_generator_state(async_module, generator_state):
    """Full roundtrip: GenerateSamplesActor produces a payload with
    generator_state → payload becomes a checkpoint → PPOTrainerAsync.fit
    restores it to the generator."""

    # Step 1: GenerateSamplesActor produces payload with generator_state
    GenActor = async_module.GenerateSamplesActor
    gen_actor = GenActor.__new__(GenActor)
    gen_actor.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1, partial_rollout_enable=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    gen_actor._partial_rollout = False
    gen_actor._last_eval_step = -1
    gen_actor._eval_just_done = False
    gen_actor.prompts_dataloader = MagicMock()
    gen_actor.prompts_dataloader.state_dict.return_value = {"cursor": 15}
    gen_actor.eval_dataloader = None

    mock_gen = MagicMock()
    mock_gen.generate_samples.return_value = (
        ["rollout-A", "rollout-B"], 0.8, 5, True
    )
    mock_gen.state_dict.return_value = generator_state
    gen_actor.samples_generator = mock_gen

    gen_actor.rollout_queue = MagicMock()
    gen_actor.rollout_slots = MagicMock()
    gen_actor.rollout_slots.get.return_value = 0
    gen_actor.vllm_lock = MagicMock()
    gen_actor.vllm_lock.acquire.remote.return_value = None
    gen_actor.vllm_lock.release.remote.return_value = None
    gen_actor.generate_kwargs = {}

    gen_actor.fit(episode=0, total_consumed_prompts=0)

    # Extract the client_states from the payload that went into the queue
    saved_client_states = None
    for put_call in gen_actor.rollout_queue.put.call_args_list:
        args, _ = put_call
        if args and isinstance(args[0], tuple) and len(args[0]) == 4:
            saved_client_states = args[0][1]
            break
    assert saved_client_states is not None
    assert saved_client_states["generator_state"] == generator_state

    # Step 2: Add global_step (what TrainingActor does) and form checkpoint
    checkpoint_from_async = dict(saved_client_states)
    checkpoint_from_async["global_step"] = 7  # training step that happened
    assert "generator_state" in checkpoint_from_async

    # Step 3: PPOTrainerAsync.fit() resumes with this checkpoint
    trainer = _build_ppotrainer_async(async_module, checkpoint_from_async)
    trainer.fit()

    call_args = trainer.generator_actor.load_state_dict.remote.call_args
    pos_args, kw_args = call_args
    assert pos_args == ({"cursor": 15},)
    assert kw_args.get("generator_state") == generator_state


# ── Symmetry audit: async client_states must match sync shape ───────────────


def _sync_expected_client_states_keys():
    """Keys that sync PPOTrainer includes in every checkpoint save."""
    return {
        "episode",
        "global_step",
        "total_consumed_prompts",
        "data_loader_state_dict",
        "generator_state",
    }


def test_async_client_states_shape_matches_sync(async_module):
    """Post-fix, the async path must produce every key the sync path produces."""
    expected_keys = _sync_expected_client_states_keys()
    GenActor = async_module.GenerateSamplesActor

    gen_actor = GenActor.__new__(GenActor)
    gen_actor.args = SimpleNamespace(
        # num_episodes must exceed the fit() episode arg so the range loop runs.
        train=SimpleNamespace(num_episodes=5, partial_rollout_enable=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    gen_actor._partial_rollout = False
    gen_actor._last_eval_step = -1
    gen_actor._eval_just_done = False
    gen_actor.prompts_dataloader = MagicMock()
    gen_actor.prompts_dataloader.state_dict.return_value = {"cursor": 0}
    gen_actor.eval_dataloader = None

    mock_gen = MagicMock()
    mock_gen.generate_samples.return_value = (["rollout-1"], 0.5, 1, True)
    mock_gen.state_dict.return_value = {"sample_buffer": [], "dataloader_exhausted": False}
    gen_actor.samples_generator = mock_gen

    gen_actor.rollout_queue = MagicMock()
    gen_actor.rollout_slots = MagicMock()
    gen_actor.rollout_slots.get.return_value = 0
    gen_actor.vllm_lock = MagicMock()
    gen_actor.vllm_lock.acquire.remote.return_value = None
    gen_actor.vllm_lock.release.remote.return_value = None
    gen_actor.generate_kwargs = {}

    # episode=3 is fine here because num_episodes=5 → range(3, 5) runs.
    gen_actor.fit(episode=3, total_consumed_prompts=10)

    saved_client_states = None
    for put_call in gen_actor.rollout_queue.put.call_args_list:
        args, _ = put_call
        if args and isinstance(args[0], tuple) and len(args[0]) == 4:
            saved_client_states = args[0][1]
            break
    assert saved_client_states is not None

    # Add global_step (TrainingActor adds this before saving checkpoint).
    # After that merge, the key set must equal the sync baseline.
    merged = dict(saved_client_states)
    merged["global_step"] = 7
    assert set(merged.keys()) == expected_keys, (
        f"Async client_states keys differ from sync: "
        f"missing={expected_keys - set(merged.keys())}, "
        f"extra={set(merged.keys()) - expected_keys}"
    )


# ── load_state_dict backward-compat with old callers ─────────────────────────


def test_load_state_dict_signature_is_backward_compatible(async_module):
    """load_state_dict(state_dict) with a single positional arg must still work
    — any existing external callers that only pass the dataloader dict must
    not break."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(partial_rollout_enable=False))
    actor.prompts_dataloader = MagicMock()
    actor.samples_generator = MagicMock()

    # Single positional call — like the pre-fix signature.
    actor.load_state_dict({"cursor": 0})

    actor.prompts_dataloader.load_state_dict.assert_called_once_with({"cursor": 0})
    # generator_state=None → samples_generator.load_state_dict NOT called.
    actor.samples_generator.load_state_dict.assert_not_called()


def test_load_state_dict_generator_state_can_be_passed_positionally(async_module, generator_state):
    """generator_state can be passed as a 2nd positional arg, matching
    the (state_dict, generator_state) convention we now use in PPOTrainerAsync.fit."""
    GenActor = async_module.GenerateSamplesActor

    actor = GenActor.__new__(GenActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(partial_rollout_enable=False))
    actor.prompts_dataloader = MagicMock()
    actor.samples_generator = MagicMock()

    actor.load_state_dict({"cursor": 0}, generator_state)

    actor.prompts_dataloader.load_state_dict.assert_called_once_with({"cursor": 0})
    actor.samples_generator.load_state_dict.assert_called_once_with(generator_state)


# ── Summary of what the audit found ─────────────────────────────────────────


def test_audit_docstrings_are_present(async_module):
    """Regression guard: the explanatory docstrings added during the audit
    must not be silently removed in future refactors."""
    import inspect

    GenActor = async_module.GenerateSamplesActor

    ld_doc = inspect.getdoc(GenActor.load_state_dict)
    assert ld_doc is not None
    assert "generator_state" in ld_doc
    assert "oversampling" in ld_doc.lower()
