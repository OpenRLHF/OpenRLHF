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


@pytest.fixture(params=["actor", "critic"])
def dynamic_ppo_worker(request, monkeypatch):
    from tests.test_experience import _experience_module
    from tests.test_loss_aggregation import _loss_module, _loss_utils_module
    from tests.test_sft_trainer import _Actor, _Engine, _load_module

    root = Path(__file__).resolve().parents[1]
    fake_ray = MagicMock()
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    # Stub distributed model loading; run native PPO training steps and losses on CPU.
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
    buffer_module = _load_module(
        monkeypatch, "_dynamic_ppo_buffer_test", root / "openrlhf/trainer/ppo_utils/replay_buffer.py"
    )
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils", buffer_module)
    kind = request.param
    module = _load_module(
        monkeypatch, f"openrlhf.trainer.ray._dynamic_test_{kind}", root / f"openrlhf/trainer/ray/ppo_{kind}.py"
    )
    cls = module.ActorPPOTrainer if kind == "actor" else module.CriticPPOTrainer
    trainer = cls.__new__(cls)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(dynamic_batch_enable=True),
        ds=SimpleNamespace(tensor_parallel_size=1),
        actor=SimpleNamespace(entropy_coef=None, aux_loss_coef=0),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=False)),
    )
    model = _Actor(1) if kind == "actor" else _Engine(1)
    engine = model.model if kind == "actor" else model
    trainer.strategy = SimpleNamespace(
        ring_attn_group=None,
        backward=lambda loss, model, optim: engine.backward(loss),
        optimizer_step=lambda *args, **kwargs: engine.step(),
        get_grad_norm=lambda _: 0.0,
    )
    trainer.aux_loss = False
    trainer.ema_model = None
    setattr(trainer, kind, model)
    setattr(trainer, f"{kind}_optim", engine.optimizer)
    setattr(trainer, f"{kind}_scheduler", engine.scheduler)
    setattr(trainer, f"{kind}_loss_fn", _loss_module.PolicyLoss() if kind == "actor" else _loss_module.ValueLoss())
    return kind, trainer, engine, _experience_module


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("token_level", [False, True])
def test_ppo_sets_dynamic_boundary_before_forward(dynamic_ppo_worker, monkeypatch, dynamic, token_level):
    kind, trainer, engine, experiences = dynamic_ppo_worker
    trainer.args.train.dynamic_batch_enable = dynamic
    engine.gas = 1 if dynamic else 3
    model = getattr(trainer, kind)
    loss_fn = getattr(trainer, f"{kind}_loss_fn")
    loss_fn.token_level_loss = token_level
    boundaries = []

    def forward(sequences, action_mask=None, **kwargs):
        boundaries.append(engine.boundary)
        return engine.weight * sequences[:, 1:].float() / 10 - 2, SimpleNamespace()

    monkeypatch.setattr(model, "forward", forward)
    # Consecutive updates have different microbatch counts; boundary state must reset.
    windows = [3, 1, 2] if dynamic else [3, 3]
    ref_weight = torch.nn.Parameter(torch.tensor(1.0))
    ref_optimizer = torch.optim.SGD([ref_weight], lr=0.01, momentum=0.9)
    ref_scheduler = torch.optim.lr_scheduler.StepLR(ref_optimizer, step_size=1, gamma=0.9)
    for count in windows:
        features = torch.arange(1, count * 3 + 1).reshape(count, 3)
        old = features.float() / 10 - 2
        batch = experiences.Experience(
            sequences=torch.cat([torch.zeros(count, 1, dtype=torch.long), features], dim=1),
            attention_mask=torch.ones(count, 4),
            action_mask=torch.arange(3)[None, :] <= torch.arange(count)[:, None],
            action_log_probs=old,
            advantages=-torch.ones(count, 3),
            values=old,
            returns=torch.zeros(count, 3),
        )
        trainer.replay_buffer = SimpleNamespace(
            dynamic_optimizer_step=[0] * (count - 1) + [1],
            dynamic_global_batch_size=[count] * count,
            dynamic_batch_num_tokens=[batch.action_mask.sum().item()] * count,
            dynamic_sample_loss_scale=[1 / count] * count,
        )
        for step, item in enumerate(experiences.split_experience_batch(batch)):
            micro = experiences.make_experience_batch([item])
            norm = (
                None
                if dynamic
                else {
                    "dp_size": 1,
                    "global_batch_size": count / engine.gas,
                    "batch_num_tokens": batch.action_mask.sum() / engine.gas,
                }
            )
            if kind == "actor":
                trainer.training_step(micro, 0.0, step, norm)
            else:
                trainer.training_step(micro, step, norm)
        values = ref_weight * features.float() / 10 - 2
        loss = (
            loss_fn(values, batch.action_log_probs, batch.advantages, batch.action_mask)[0]
            if kind == "actor"
            else loss_fn(values, batch.values, batch.returns, batch.action_mask)
        )
        loss.backward()
        ref_optimizer.step()
        ref_optimizer.zero_grad()
        ref_scheduler.step()
        torch.testing.assert_close(engine.weight, ref_weight)
        torch.testing.assert_close(
            engine.optimizer.state[engine.weight]["momentum_buffer"],
            ref_optimizer.state[ref_weight]["momentum_buffer"],
        )
    assert boundaries == ([False, False, True, True, False, True] if dynamic else [None] * 6)
    assert engine.global_steps == len(windows)
    assert engine.scheduler.state_dict() == ref_scheduler.state_dict()
