import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _Pbar:
    def __init__(self, iterable, **_):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, _):
        pass

    def set_postfix(self, _):
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


class _PPOModel(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.model = engine
        self.seen = []

    def forward(self, sequences, action_mask, **kwargs):
        self.seen.extend(sequences[:, 0].tolist())
        return self.model.weight * sequences[:, 1:].float() / 10 - 2, SimpleNamespace()


@pytest.fixture(params=["actor", "critic"])
def ppo_worker(request, monkeypatch):
    from tests.test_experience import _experience_module
    from tests.test_loss_aggregation import _loss_module, _loss_utils_module
    from tests.test_sft_trainer import _Engine, _load_module

    root = Path(__file__).resolve().parents[1]
    fake_ray = MagicMock()
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    # Isolate Ray/model loading; execute the real PPO loops, losses and replay collation on CPU.
    for name in (
        "openrlhf.trainer.ray.launcher",
        "openrlhf.trainer.ray.utils",
        "openrlhf.utils",
        "openrlhf.utils.deepspeed",
        "openrlhf.utils.deepspeed.deepspeed_utils",
        "openrlhf.utils.distributed_util",
        "openrlhf.utils.vlm_utils",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", SimpleNamespace(init_logger=logging.getLogger))
    monkeypatch.setitem(sys.modules, "openrlhf.utils.loss_utils", _loss_utils_module)
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", _experience_module)
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.models",
        SimpleNamespace(
            Actor=None,
            PolicyLoss=_loss_module.PolicyLoss,
            ValueLoss=_loss_module.ValueLoss,
            aggregate_loss=_loss_module.aggregate_loss,
            get_llm_for_sequence_regression=None,
        ),
    )
    monkeypatch.setitem(sys.modules, "openrlhf.models.utils", sys.modules[_loss_module.__package__ + ".utils"])
    _load_module(monkeypatch, "openrlhf.utils.seqlen_balancing", root / "openrlhf/utils/seqlen_balancing.py")
    buffer_module = _load_module(monkeypatch, "_ppo_buffer_test", root / "openrlhf/trainer/ppo_utils/replay_buffer.py")
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils", buffer_module)
    kind = request.param
    module = _load_module(
        monkeypatch, f"openrlhf.trainer.ray._test_{kind}", root / f"openrlhf/trainer/ray/ppo_{kind}.py"
    )
    monkeypatch.setattr(module, "tqdm", _Pbar)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    engine = _Engine(4)
    model = _PPOModel(engine)
    cls = module.ActorPPOTrainer if kind == "actor" else module.CriticPPOTrainer
    trainer = cls.__new__(cls)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(dynamic_batch_enable=False),
        ds=SimpleNamespace(tensor_parallel_size=1),
        actor=SimpleNamespace(entropy_coef=None, aux_loss_coef=0),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=False)),
    )
    trainer.strategy = SimpleNamespace(
        accumulated_gradient=4,
        ring_attn_group=None,
        is_rank_0=lambda: False,
        all_reduce=lambda value: value,
        all_gather=MagicMock(side_effect=lambda n: torch.tensor([n])),
        backward=lambda loss, model, optim: engine.backward(loss),
        optimizer_step=lambda *args, **kwargs: engine.step(),
        get_grad_norm=lambda _: 0.0,
    )
    trainer.max_epochs = 1
    trainer.dataloader_pin_memory = trainer.aux_loss = False
    trainer.ema_model = None
    setattr(trainer, kind, model)
    setattr(trainer, f"{kind}_optim", engine.optimizer)
    setattr(trainer, f"{kind}_scheduler", engine.scheduler)
    setattr(trainer, f"{kind}_loss_fn", _loss_module.PolicyLoss() if kind == "actor" else _loss_module.ValueLoss())
    buffer = buffer_module.NaiveReplayBuffer.__new__(buffer_module.NaiveReplayBuffer)
    buffer.sample_batch_size = 1
    buffer.dynamic_batch = buffer.packing_samples = False
    buffer.items = []
    trainer.replay_buffer = buffer
    return kind, trainer, model, _experience_module


def _ppo_samples(experiences, count):
    ids = torch.arange(count)
    features = torch.stack([ids + 1, ids + 2, ids + 3], dim=1)
    old = features.float() / 10 - 2
    batch = experiences.Experience(
        sequences=torch.cat([ids[:, None], features], dim=1),
        attention_mask=torch.ones(count, 4),
        action_mask=torch.arange(3)[None, :] <= (ids % 3)[:, None],
        action_log_probs=old,
        advantages=-torch.ones(count, 3),
        values=old,
        returns=torch.zeros(count, 3),
    )
    return experiences.split_experience_batch(batch)


@pytest.mark.parametrize("token_level", [False, True])
@pytest.mark.parametrize(
    "gas,micro,count,epochs,shuffle",
    [
        (4, 1, 5, 2, False),
        (4, 2, 11, 2, True),
        (4, 1, 8, 1, False),
        (1, 2, 5, 2, True),
        (4, 1, 3, 2, False),
        (4, 1, 0, 1, True),
    ],
)
def test_ppo_complete_windows_match_full_batch(ppo_worker, token_level, gas, micro, count, epochs, shuffle, caplog):
    kind, trainer, model, experiences = ppo_worker
    engine = model.model
    engine.gas = trainer.strategy.accumulated_gradient = gas
    trainer.max_epochs = epochs
    trainer.args.ds.tensor_parallel_size = 1 if shuffle else 2
    trainer.replay_buffer.sample_batch_size = micro
    loss_fn = getattr(trainer, f"{kind}_loss_fn")
    loss_fn.token_level_loss = token_level
    reference = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([reference], lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    expected_steps = 0
    torch.manual_seed(17)
    # Keep the same optimizer across calls, just as consecutive PPO rollouts do.
    for size in (count, 8):
        samples = _ppo_samples(experiences, size)
        trainer.replay_buffer.items = samples
        model.seen.clear()
        caplog.clear()
        status = trainer.ppo_train(0) if kind == "actor" else trainer.ppo_train()
        used = size // (gas * micro) * gas * micro
        assert len(model.seen) == used * epochs
        assert bool(status) == bool(used)
        assert engine.weight.grad is None
        assert engine.micro_steps % gas == 0
        if size > used:
            assert f"Skipping {size - used} {kind} samples per epoch on this rank" in caplog.text
        else:
            assert "Skipping" not in caplog.text
        for epoch in range(epochs):
            seen = model.seen[epoch * used : (epoch + 1) * used]
            assert len(set(seen)) == used
            if not shuffle:
                assert seen == list(range(used))
            for start in range(0, used, gas * micro):
                batch = experiences.make_experience_batch([samples[i] for i in seen[start : start + gas * micro]])
                values = reference * batch.sequences[:, 1:].float() / 10 - 2
                if kind == "actor":
                    loss = loss_fn(values, batch.action_log_probs, batch.advantages, batch.action_mask)[0]
                else:
                    loss = loss_fn(values, batch.values, batch.returns, batch.action_mask)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                expected_steps += 1
        assert engine.global_steps == expected_steps
        torch.testing.assert_close(engine.weight, reference)
        if expected_steps:
            torch.testing.assert_close(
                engine.optimizer.state[engine.weight]["momentum_buffer"], optimizer.state[reference]["momentum_buffer"]
            )
        assert engine.scheduler.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize("peer_steps", [0, 1])
def test_ppo_limits_updates_to_smallest_rank(ppo_worker, peer_steps):
    kind, trainer, model, experiences = ppo_worker
    trainer.replay_buffer.items = _ppo_samples(experiences, 9)
    trainer.strategy.all_gather.side_effect = lambda n: torch.tensor([n, peer_steps])
    trainer.ppo_train(0) if kind == "actor" else trainer.ppo_train()
    trainer.strategy.all_gather.assert_called_once_with(2)
    assert len(model.seen) == peer_steps * 4
    assert model.model.global_steps == peer_steps
    assert model.model.weight.grad is None


def test_ppo_dynamic_batch_keeps_its_optimizer_boundaries(ppo_worker):
    kind, trainer, model, experiences = ppo_worker
    trainer.args.train.dynamic_batch_enable = True
    model.model.gas = trainer.strategy.accumulated_gradient = 1
    buffer = trainer.replay_buffer
    buffer.dynamic_batch = True
    buffer.items = _ppo_samples(experiences, 3)
    buffer.dynamic_indices = [[0], [1], [2]]
    buffer.dynamic_batch_num_tokens = [6] * 3
    buffer.dynamic_global_batch_size = [3] * 3
    buffer.dynamic_sample_loss_scale = [1 / 3] * 3
    buffer.dynamic_optimizer_step = [False, False, True]
    buffer.setup_dynamic_batch = MagicMock()
    trainer.ppo_train(0) if kind == "actor" else trainer.ppo_train()
    buffer.setup_dynamic_batch.assert_called_once_with(trainer.strategy)
    trainer.strategy.all_gather.assert_not_called()
    assert model.seen == [0, 1, 2]
    assert model.model.global_steps == 1
    assert model.model.weight.grad is None
