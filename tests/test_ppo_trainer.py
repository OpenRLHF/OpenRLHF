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


def _checkpoint_trainer(module):
    trainer = module.BasePPOTrainer.__new__(module.BasePPOTrainer)
    trainer.args = SimpleNamespace(logger=SimpleNamespace(logging_steps=1), ckpt=SimpleNamespace(save_steps=1))
    trainer.actor_model_group = MagicMock()
    trainer.actor_model_group.async_run_method.return_value = []
    trainer.critic_model_group = None
    trainer.wandb_logger = trainer.tensorboard_logger = None
    trainer.best_eval_metric_key = ""
    trainer.best_eval_metric_value = float("-inf")
    trainer._latest_eval_metric_value = None
    return trainer


@pytest.mark.parametrize("best,latest", [(0.8, 0.5), (0.0, -0.2), (-0.2, -0.5)])
def test_regular_checkpoint_restores_best_and_latest_metrics(ppo_module, tmp_path, best, latest):
    import torch

    trainer = _checkpoint_trainer(ppo_module)
    trainer.save_best_checkpoint({"eval_math_pass1": best}, 1)
    trainer.save_best_checkpoint({"eval_math_pass1": latest}, 2)
    trainer.save_logs_and_checkpoints(3, client_states={"global_step": 3})
    saved = trainer.actor_model_group.async_run_method.call_args.kwargs["client_states"]
    path = tmp_path / "client_state.pt"
    torch.save(saved, path)

    resumed = _checkpoint_trainer(ppo_module)
    resumed.restore_best_checkpoint_state(torch.load(path, weights_only=True))
    assert resumed.best_eval_metric_key == "eval_math_pass1"
    assert resumed.best_eval_metric_value == best
    assert resumed._latest_eval_metric_value == latest
    resumed.save_best_checkpoint({"eval_math_pass1": (best + latest) / 2}, 4)
    resumed.actor_model_group.async_run_method.assert_not_called()
    resumed.save_best_checkpoint({"eval_math_pass1": best + 0.1}, 5)
    assert resumed.actor_model_group.async_run_method.call_args.kwargs["tag"] == "best_global_step5"


def test_best_checkpoint_overwrites_previous_latest_metric(ppo_module):
    trainer = _checkpoint_trainer(ppo_module)
    trainer.save_best_checkpoint({"eval_math_pass1": 0.8}, 1, {"latest_eval_metric_value": 0.2})
    saved = trainer.actor_model_group.async_run_method.call_args.kwargs["client_states"]
    resumed = _checkpoint_trainer(ppo_module)
    resumed.restore_best_checkpoint_state(saved)
    assert resumed.best_eval_metric_value == resumed._latest_eval_metric_value == 0.8


@pytest.mark.parametrize(
    "state,best,latest",
    [
        ({}, float("-inf"), None),
        ({"best_eval_metric_key": "eval_math_pass1", "best_eval_metric_value": 0.0}, 0.0, 0.0),
        ({"best_eval_metric_value": 0.8, "latest_eval_metric_value": None}, 0.8, None),
    ],
)
def test_restore_checkpoint_metric_compatibility(ppo_module, state, best, latest):
    trainer = _checkpoint_trainer(ppo_module)
    trainer.restore_best_checkpoint_state(state)
    assert trainer.best_eval_metric_value == best
    assert trainer._latest_eval_metric_value == latest


@pytest.fixture
def async_ppo_module(ppo_module, monkeypatch):
    fake_ray = sys.modules["ray"]
    fake_ray.method.side_effect = lambda **_: lambda fn: fn
    monkeypatch.setitem(sys.modules, "ray.util.queue", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", ppo_module)
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_async_ppo_test",
        Path(__file__).resolve().parents[1] / "openrlhf/trainer/ppo_trainer_async.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("start,interval", [(0, 1), (0, 2), (10, 3)])
def test_async_best_checkpoint_evaluates_exact_updated_model(async_ppo_module, start, interval):
    from queue import Queue

    trainer = async_ppo_module.TrainingActor.__new__(async_ppo_module.TrainingActor)
    trainer.args = SimpleNamespace(
        eval=SimpleNamespace(dataset="math", steps=interval),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    trainer.rollout_queue, trainer.rollout_slots = Queue(), Queue()
    for sample in range(1, 7):
        trainer.rollout_queue.put(([sample], {"total_consumed_prompts": sample}, None, 0))
    trainer.rollout_queue.put("done")
    trainer.wandb_logger = trainer.tensorboard_logger = None
    trainer.best_eval_metric_key, trainer.best_eval_metric_value = "", float("-inf")
    trainer.critic_model_group = None
    events, state = [], {"step": start}

    def train(samples, step):
        assert step == state["step"]
        state["step"] += 1
        events.append(("train", state["step"]))
        return {}, state["step"]

    def evaluate():
        events.append(("eval", state["step"]))
        return {"eval_math_pass1": state["step"] / 100}

    def save(**kwargs):
        assert kwargs["method_name"] == "save_checkpoint"
        assert kwargs["tag"] == f"best_global_step{state['step']}"
        assert kwargs["client_states"]["global_step"] == state["step"]
        assert kwargs["client_states"]["total_consumed_prompts"] == state["step"] - start
        assert kwargs["metric_value"] == state["step"] / 100
        events.append(("save", state["step"]))
        return []

    trainer.train_step = train
    trainer.generator_actor = SimpleNamespace(_run_eval=SimpleNamespace(remote=evaluate))
    trainer.actor_model_group = SimpleNamespace(async_run_method=save)
    trainer.save_logs_and_checkpoints = lambda *_: None
    trainer.fit(start)
    expected = []
    for step in range(start + 1, start + 7):
        expected.append(("train", step))
        if step % interval == 0:
            expected.extend([("eval", step), ("save", step)])
    assert events == expected
    assert trainer.rollout_slots.qsize() == 6


@pytest.mark.parametrize("best_key,dataset", [("none", "math"), ("", None)])
def test_async_trainer_does_not_request_unneeded_eval(async_ppo_module, best_key, dataset):
    from queue import Queue

    trainer = async_ppo_module.TrainingActor.__new__(async_ppo_module.TrainingActor)
    trainer.args = SimpleNamespace(
        eval=SimpleNamespace(dataset=dataset, steps=1), algo=SimpleNamespace(dynamic_filtering_enable=False)
    )
    trainer.rollout_queue, trainer.rollout_slots = Queue(), Queue()
    trainer.rollout_queue.put(([1], {}, None, 0))
    trainer.rollout_queue.put("done")
    trainer.best_eval_metric_key = best_key
    trainer.wandb_logger = trainer.tensorboard_logger = None
    trainer.generator_actor = MagicMock()
    trainer.train_step = lambda _samples, step: ({}, step + 1)
    trainer.save_logs_and_checkpoints = lambda *_: None
    trainer.fit()
    trainer.generator_actor._run_eval.remote.assert_not_called()


@pytest.mark.parametrize("best_key,expected", [("", False), ("eval_math_pass1", False), ("none", True)])
def test_async_generator_owns_only_eval_without_best_saving(async_ppo_module, best_key, expected):
    generator = async_ppo_module.GenerateSamplesActor.__new__(async_ppo_module.GenerateSamplesActor)
    generator.args = SimpleNamespace(ckpt=SimpleNamespace(best_metric_key=best_key), eval=SimpleNamespace(steps=2))
    generator.eval_dataloader = object()
    generator._last_eval_step = -1
    generator._best_metric_key = best_key
    assert generator._should_eval(2) is expected


def test_async_eval_releases_both_locks_on_failure(async_ppo_module):
    import threading

    generator = async_ppo_module.GenerateSamplesActor.__new__(async_ppo_module.GenerateSamplesActor)
    generator._generation_lock = threading.Lock()
    generator.vllm_lock = MagicMock()
    generator.generate_kwargs = {}
    generator.args = SimpleNamespace(eval=SimpleNamespace(temperature=1.0, n_samples_per_prompt=1))
    generator.eval_dataloader = []
    generator.samples_generator = MagicMock()
    generator.samples_generator.generate_eval_samples.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        generator._run_eval()
    assert generator._generation_lock.acquire(blocking=False)
    generator._generation_lock.release()
    generator.vllm_lock.release.remote.assert_called_once()
    generator.samples_generator.generate_eval_samples.side_effect = None
    generator.samples_generator.generate_eval_samples.return_value = []
    assert generator._run_eval() == {}


@pytest.mark.parametrize("configured,restored", [("", "none"), ("none", "eval_math_pass1"), ("none", None)])
def test_async_resume_agrees_on_eval_owner(async_ppo_module, configured, restored):
    from queue import Queue

    state = {"episode": 0, "global_step": 2, "data_loader_state_dict": {}, "best_eval_metric_key": restored}
    coordinator = async_ppo_module.PPOTrainerAsync.__new__(async_ppo_module.PPOTrainerAsync)
    coordinator.trainer_actor = MagicMock()
    coordinator.generator_actor = MagicMock()
    coordinator.trainer_actor.init_checkpoint_states.remote.return_value = state
    coordinator.fit()
    call = coordinator.generator_actor.fit.remote.call_args.kwargs
    assert call["best_metric_key"] == restored

    generator = async_ppo_module.GenerateSamplesActor.__new__(async_ppo_module.GenerateSamplesActor)
    generator._best_metric_key = configured
    generator.args = SimpleNamespace(train=SimpleNamespace(num_episodes=0), eval=SimpleNamespace(steps=2))
    generator.rollout_queue = Queue()
    generator.eval_dataloader = object()
    generator._last_eval_step = -1
    generator.fit(**call)
    trainer = async_ppo_module.TrainingActor.__new__(async_ppo_module.TrainingActor)
    trainer.best_eval_metric_key = configured
    trainer.restore_best_checkpoint_state(state)
    assert generator._should_eval(2) == (trainer.best_eval_metric_key == "none")
