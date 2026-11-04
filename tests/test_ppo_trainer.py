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


@pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("n_samples", [1, 2, 4])
@pytest.mark.parametrize("all_zero", [False, True])
def test_eval_metrics_accept_integer_rewards(ppo_module, dtype, n_samples, all_zero):
    import torch

    values = [0 if all_zero else i % 2 for i in range(3 * n_samples)]
    samples = [
        SimpleNamespace(
            prompts=[f"p{i // n_samples}"],
            rewards=torch.tensor([value], dtype=getattr(torch, dtype)),
            response_length=torch.tensor([i + 1]),
            truncated=torch.tensor([all_zero]),
        )
        for i, value in enumerate(values)
    ]
    loader = [(["math", "math", "code"], ["p0", "p1", "p2"], ["", "", ""], [None] * 3)]
    logs = ppo_module.compute_eval_metrics(loader, samples, n_samples)
    for datasource, start, end in [("math", 0, 2 * n_samples), ("code", 2 * n_samples, 3 * n_samples)]:
        assert logs[f"eval_{datasource}_pass1"] == sum(values[start:end]) / (end - start)
        expected_pass_k = sum(max(values[i : i + n_samples]) for i in range(start, end, n_samples))
        assert logs[f"eval_{datasource}_pass{n_samples}"] == expected_pass_k / ((end - start) / n_samples)
        assert logs[f"eval_{datasource}_response_length_mean"] == (start + 1 + end) / 2
        assert logs[f"eval_{datasource}_truncated_rate"] == float(all_zero)
    assert logs["eval_num_samples"] == len(samples)
    assert logs["eval_response_length_mean"] == (1 + len(samples)) / 2
    assert logs["eval_truncated_rate"] == float(all_zero)
    assert all(sample.rewards.dtype == getattr(torch, dtype) for sample in samples)


@pytest.mark.parametrize(
    "dtypes,values",
    [
        (["int64", "float32"], [0, 0.75]),
        (["float16"] * 3, [0.1, 0.2, 0.4]),
        (["bfloat16"] * 3, [0.1, 0.2, 0.4]),
        (["float32"] * 3, [-0.5, 0.1, 0.8]),
        (["float64"] * 3, [1.00000003, -1.0, 0.0]),
    ],
)
def test_eval_metrics_preserve_floating_rewards(ppo_module, dtypes, values):
    import torch

    rewards = [torch.tensor([value], dtype=getattr(torch, dtype)) for dtype, value in zip(dtypes, values)]
    samples = [SimpleNamespace(prompts=["p"], rewards=r, response_length=None, truncated=None) for r in rewards]
    logs = ppo_module.compute_eval_metrics([(["math"], ["p"], [""], [None])], samples, len(samples))
    original = torch.tensor(rewards)
    assert logs == {
        f"eval_math_pass{len(samples)}": original.max().float().item(),
        "eval_math_pass1": original.mean().float().item(),
        "eval_num_samples": float(len(samples)),
    }


def test_eval_metrics_empty_samples(ppo_module):
    assert ppo_module.compute_eval_metrics([], [], 1) == {}
