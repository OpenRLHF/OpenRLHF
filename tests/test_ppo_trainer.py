import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _Pbar:
    def __init__(self, iterable, **_):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, _):
        pass

    def close(self):
        pass


class _ObjectRef:
    def __init__(self, value):
        self.value = tuple(value)


@pytest.fixture
def ppo_module(monkeypatch):
    fake_ray = MagicMock()
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj
    fake_ray.get.side_effect = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=_Pbar))

    # Keep this control-flow test importable without Ray, vLLM, DeepSpeed, or datasets.
    for name in (
        "openrlhf.datasets",
        "openrlhf.datasets.utils",
        "openrlhf.trainer.ppo_utils.experience",
        "openrlhf.trainer.ppo_utils.experience_maker",
        "openrlhf.trainer.ppo_utils.kl_controller",
        "openrlhf.trainer.ppo_utils.samples_generator",
        "openrlhf.trainer.ray.launcher",
        "openrlhf.trainer.ray.vllm_engine",
        "openrlhf.utils.deepspeed",
        "openrlhf.utils.logging_utils",
        "openrlhf.utils.utils",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_ppo_trainer_test",
        root / "openrlhf" / "trainer" / "ppo_trainer.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.tqdm = _Pbar
    return module


@pytest.fixture
def ppo_async_module(ppo_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "ray.util.queue", SimpleNamespace(Queue=MagicMock()))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", ppo_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_ppo_trainer_async_test",
        root / "openrlhf" / "trainer" / "ppo_trainer_async.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.tqdm = _Pbar
    module.ray.put.side_effect = lambda value: _ObjectRef(value)
    module.ray.get.side_effect = lambda value: list(value.value) if isinstance(value, _ObjectRef) else value
    return module


class _Loader:
    def __len__(self):
        return 129

    def state_dict(self):
        return {"done": True}


class _Generator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def generate_samples(self, **_):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("The terminal result must end the episode")
        return self.result


def _trainer(module, result):
    trainer = module.PPOTrainer.__new__(module.PPOTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    trainer.prompts_dataloader = _Loader()
    trainer.eval_dataloader = None
    trainer.samples_generator = _Generator(result)
    trainer.generate_kwargs = {}
    trainer.init_checkpoint_states = lambda: {
        "episode": 0,
        "global_step": 0,
        "total_consumed_prompts": 0,
        "data_loader_state_dict": {},
    }
    trainer.restore_best_checkpoint_state = lambda _: None
    trainer.save_logs_and_checkpoints = lambda *_: None
    trainer.wandb_logger = trainer.tensorboard_logger = None
    return trainer


def test_sync_fit_processes_nonempty_exhausted_rollout(ppo_module):
    tail = [f"rollout-{i}" for i in range(129)]
    trainer = _trainer(ppo_module, (tail, None, 129, True))
    calls = []
    trainer.train_step = lambda samples, step: (calls.append(samples) or ({}, step + 1))

    trainer.fit()

    assert calls == [tail]
    assert trainer.samples_generator.calls == 1


def test_sync_fit_does_not_train_empty_exhausted_result(ppo_module):
    trainer = _trainer(ppo_module, ([], None, 0, True))
    trainer.train_step = lambda *_: pytest.fail("empty terminal result must not be trained")

    trainer.fit()

    assert trainer.samples_generator.calls == 1


def test_async_checkpoint_restores_oversampling_buffer(ppo_async_module):
    actor = ppo_async_module.GenerateSamplesActor.__new__(ppo_async_module.GenerateSamplesActor)
    actor.prompts_dataloader = MagicMock()
    actor.prompts_dataloader.__iter__.return_value = iter([4, 5])
    actor.samples_generator = SimpleNamespace()

    actor.load_state_dict({"position": 4}, [2, 3], False)

    actor.prompts_dataloader.load_state_dict.assert_called_once_with({"position": 4})
    assert actor.samples_generator._sample_buffer == [2, 3]
    assert next(actor.samples_generator._dataloader_iter) == 4
    assert not actor._dataloader_exhausted
    assert ppo_async_module.ray.get(actor._sample_buffer_ref) == [2, 3]

    legacy_actor = ppo_async_module.GenerateSamplesActor.__new__(ppo_async_module.GenerateSamplesActor)
    legacy_actor.prompts_dataloader = MagicMock()
    legacy_actor.prompts_dataloader.__iter__.return_value = iter([4, 5])
    legacy_actor.samples_generator = SimpleNamespace()
    legacy_actor.load_state_dict({"position": 4})

    assert legacy_actor.samples_generator._sample_buffer == []
    assert not legacy_actor._dataloader_exhausted
    assert legacy_actor._sample_buffer_ref is None


def test_async_client_states_share_one_buffer_snapshot(ppo_async_module):
    samples_generator = SimpleNamespace(_sample_buffer=[], _dataloader_iter=object(), calls=0)

    def generate_samples(**_):
        samples_generator.calls += 1
        if samples_generator.calls == 1:
            samples_generator._sample_buffer = [2, 3, 4, 5]
            return [0, 1], None, 3, False
        if samples_generator.calls == 2:
            samples_generator._sample_buffer = [4, 5]
            return [2, 3], None, 0, False
        if samples_generator.calls == 3:
            samples_generator._sample_buffer = []
            return [4, 5], None, 0, False
        samples_generator._dataloader_iter = None
        return [], None, 0, True

    samples_generator.generate_samples = generate_samples
    actor = ppo_async_module.GenerateSamplesActor.__new__(ppo_async_module.GenerateSamplesActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(num_episodes=1))
    actor.prompts_dataloader = MagicMock()
    actor.prompts_dataloader.__len__.return_value = 3
    actor.prompts_dataloader.state_dict.return_value = {"position": 3}
    actor.eval_dataloader = None
    actor.samples_generator = samples_generator
    actor.generate_kwargs = {}
    actor._partial_rollout = True
    actor.rollout_slots = MagicMock()
    actor.rollout_slots.get.return_value = 0
    actor.rollout_queue = MagicMock()
    actor._sample_buffer_ref = None
    actor._dataloader_exhausted = False
    actor._last_eval_step = -1
    actor._eval_just_done = False

    actor.fit(episode=0, total_consumed_prompts=0)

    payloads = [call.args[0] for call in actor.rollout_queue.put.call_args_list if isinstance(call.args[0], tuple)]
    states = [payload[1] for payload in payloads]

    assert len(states) == 3
    assert all(state["dataloader_exhausted"] for state in states)
    first_ref, first_size = states[0]["sample_buffer_state"]
    second_ref, second_size = states[1]["sample_buffer_state"]
    assert first_ref is second_ref
    assert ppo_async_module.ray.get(first_ref)[-first_size:] == [2, 3, 4, 5]
    assert ppo_async_module.ray.get(second_ref)[-second_size:] == [4, 5]
    assert states[2]["sample_buffer_state"] is None
    assert ppo_async_module.ray.put.call_count == 1


@pytest.mark.parametrize(
    ("buffer", "expected_batches"),
    [
        ([], []),
        ([2], [[2]]),
        ([2, 3], [[2, 3]]),
        ([2, 3, 4, 5], [[2, 3], [4, 5]]),
    ],
)
def test_async_exhausted_state_survives_buffer_drain(ppo_async_module, buffer, expected_batches):
    samples_generator = SimpleNamespace()

    def generate_samples(**_):
        chunk_size = 2
        if len(samples_generator._sample_buffer) < chunk_size and samples_generator._dataloader_iter is not None:
            try:
                next(samples_generator._dataloader_iter)
            except StopIteration:
                samples_generator._dataloader_iter = None
        rollout_samples = samples_generator._sample_buffer[:chunk_size]
        samples_generator._sample_buffer = samples_generator._sample_buffer[chunk_size:]
        exhausted = samples_generator._dataloader_iter is None and not samples_generator._sample_buffer
        return rollout_samples, None, 0, exhausted

    samples_generator.generate_samples = generate_samples
    actor = ppo_async_module.GenerateSamplesActor.__new__(ppo_async_module.GenerateSamplesActor)
    actor.args = SimpleNamespace(train=SimpleNamespace(num_episodes=1))
    actor.prompts_dataloader = MagicMock()
    actor.prompts_dataloader.__len__.return_value = 4
    actor.prompts_dataloader.state_dict.return_value = {"finished": True}
    actor.eval_dataloader = None
    actor.samples_generator = samples_generator
    actor.generate_kwargs = {}
    actor._partial_rollout = True
    actor.rollout_slots = MagicMock()
    actor.rollout_slots.get.return_value = 0
    actor.rollout_queue = MagicMock()
    actor._last_eval_step = -1
    actor._eval_just_done = False

    actor.load_state_dict({"finished": True}, buffer, True)
    actor.fit(episode=0, total_consumed_prompts=4)

    payloads = [call.args[0] for call in actor.rollout_queue.put.call_args_list if isinstance(call.args[0], tuple)]
    assert [payload[0] for payload in payloads] == expected_batches
    for index, (_, state, _, _) in enumerate(payloads):
        remaining = buffer[(index + 1) * 2 :]
        assert state["dataloader_exhausted"] is True
        if remaining:
            sample_buffer_ref, size = state["sample_buffer_state"]
            assert ppo_async_module.ray.get(sample_buffer_ref)[-size:] == remaining
        else:
            assert state["sample_buffer_state"] is None
    assert ppo_async_module.ray.put.call_count == bool(buffer)


def test_async_buffer_is_materialized_only_for_checkpoint(ppo_async_module):
    buffer_ref = _ObjectRef([2, 3, 4, 5])
    first_state = {"sample_buffer_state": (buffer_ref, 4)}
    second_state = {"sample_buffer_state": (buffer_ref, 2)}
    payloads = [
        (["rollout-1"], first_state, None, 0.1),
        (["rollout-2"], second_state, None, 0.1),
    ]
    actor = ppo_async_module.TrainingActor.__new__(ppo_async_module.TrainingActor)
    actor.args = SimpleNamespace(
        algo=SimpleNamespace(dynamic_filtering_enable=False), ckpt=SimpleNamespace(save_steps=2)
    )
    actor.rollout_queue = MagicMock()
    actor.rollout_queue.get.side_effect = [*payloads, "done"]
    actor.rollout_slots = MagicMock()
    actor.train_step = MagicMock(side_effect=lambda samples, step: ({"samples": samples}, step + 1))
    actor.save_logs_and_checkpoints = MagicMock()
    actor.save_best_checkpoint = MagicMock()
    actor.wandb_logger = None
    actor.tensorboard_logger = None

    actor.fit(global_step=0)

    first_saved_state = actor.save_logs_and_checkpoints.call_args_list[0].args[2]
    second_saved_state = actor.save_logs_and_checkpoints.call_args_list[1].args[2]
    assert first_saved_state["sample_buffer_state"] == (buffer_ref, 4)
    assert "sample_buffer" not in first_saved_state
    assert second_saved_state["sample_buffer"] == [4, 5]
    assert "sample_buffer_state" not in second_saved_state


def test_async_buffer_is_materialized_for_best_checkpoint(ppo_async_module):
    buffer_ref = _ObjectRef([2, 3, 4, 5])
    client_state = {"sample_buffer_state": (buffer_ref, 2)}
    payloads = [
        (["rollout"], client_state, None, 0.1),
        ("eval", 1, {"reward": 1.0}),
    ]
    actor = ppo_async_module.TrainingActor.__new__(ppo_async_module.TrainingActor)
    actor.args = SimpleNamespace(
        algo=SimpleNamespace(dynamic_filtering_enable=False), ckpt=SimpleNamespace(save_steps=float("inf"))
    )
    actor.rollout_queue = MagicMock()
    actor.rollout_queue.get.side_effect = [*payloads, "done"]
    actor.rollout_slots = MagicMock()
    actor.train_step = MagicMock(side_effect=lambda samples, step: ({"samples": samples}, step + 1))
    actor.save_logs_and_checkpoints = MagicMock()
    actor.wandb_logger = None
    actor.tensorboard_logger = None
    actor.best_eval_metric_key = "reward"
    actor.best_eval_metric_value = float("-inf")
    actor.actor_model_group = MagicMock()
    actor.actor_model_group.async_run_method.return_value = []
    actor.critic_model_group = None

    actor.fit(global_step=0)

    best_state = actor.actor_model_group.async_run_method.call_args.kwargs["client_states"]
    assert best_state["sample_buffer"] == [4, 5]
    assert "sample_buffer_state" not in best_state
