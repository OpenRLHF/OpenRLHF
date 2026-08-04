import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_ppo_trainer_module(monkeypatch):
    ray_module = types.ModuleType("ray")
    ray_module.get = lambda value: value
    ray_module.remote = lambda obj=None, **kwargs: obj if obj is not None else lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray_module)

    stubs = {
        "openrlhf.datasets": {"PromptDataset": object},
        "openrlhf.datasets.utils": {"blending_datasets": lambda *args, **kwargs: None},
        "openrlhf.trainer.ppo_utils.experience": {"balance_experiences": lambda value, args: value},
        "openrlhf.trainer.ppo_utils.experience_maker": {"RemoteExperienceMaker": object},
        "openrlhf.trainer.ppo_utils.kl_controller": {
            "AdaptiveKLController": object,
            "FixedKLController": object,
        },
        "openrlhf.trainer.ppo_utils.samples_generator": {"SamplesGenerator": object},
        "openrlhf.trainer.ray.launcher": {"RayActorGroup": object},
        "openrlhf.trainer.ray.vllm_engine": {"batch_vllm_engine_call": lambda *args, **kwargs: None},
        "openrlhf.utils.deepspeed": {"DeepspeedStrategy": object},
        "openrlhf.utils.logging_utils": {
            "TensorboardLogger": object,
            "WandbLogger": object,
            "init_logger": logging.getLogger,
        },
        "openrlhf.utils.utils": {"get_tokenizer": lambda *args, **kwargs: None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.trainer.ppo_trainer", root / "openrlhf" / "trainer" / "ppo_trainer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_checkpoint_states_fills_legacy_controller_fields(monkeypatch):
    module = _load_ppo_trainer_module(monkeypatch)

    states = module.normalize_checkpoint_states({"episode": 2, "global_step": 7})

    assert states == {
        "episode": 2,
        "global_step": 7,
        "total_consumed_prompts": 0,
        "data_loader_state_dict": {},
    }


def test_init_checkpoint_states_normalizes_loaded_state(monkeypatch, tmp_path):
    module = _load_ppo_trainer_module(monkeypatch)
    trainer = module.BasePPOTrainer.__new__(module.BasePPOTrainer)
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(path=str(tmp_path), load_enable=True))
    (tmp_path / "_actor").mkdir()
    trainer.actor_model_group = SimpleNamespace(async_run_method=lambda **kwargs: [{"episode": 1, "global_step": 3}])

    states = trainer.init_checkpoint_states()

    assert states["episode"] == 1
    assert states["global_step"] == 3
    assert states["total_consumed_prompts"] == 0
    assert states["data_loader_state_dict"] == {}
