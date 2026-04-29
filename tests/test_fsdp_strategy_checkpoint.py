import torch.nn as nn

from openrlhf.utils.fsdp.strategy import FsdpStrategy


class _BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"normalize_reward": False})()


class _Wrapper(nn.Module):
    def __init__(self, peft_config):
        super().__init__()
        self.base = _BaseModel()
        self.peft_config = peft_config

    def get_base_model_for_fsdp(self):
        return self.base


def test_save_ckpt_forwards_wrapper_peft_config(tmp_path):
    strategy = object.__new__(FsdpStrategy)
    calls = []

    class FakeCheckpointer:
        def save_model(self, **kwargs):
            calls.append(kwargs)
            model_dir = tmp_path / "global_step1" / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

    strategy._build_checkpointer = lambda *args, **kwargs: FakeCheckpointer()
    peft_config = object()

    strategy.save_ckpt(
        _Wrapper(peft_config),
        str(tmp_path),
        "global_step1",
        max_num=3,
        max_mem=0,
    )

    assert calls[0]["peft_config"] is peft_config


def test_best_ckpt_does_not_overwrite_latest(tmp_path):
    strategy = object.__new__(FsdpStrategy)

    class FakeCheckpointer:
        def save_model(self, **kwargs):
            model_dir = tmp_path / kwargs["weights_path"].split("/")[-1] / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

    strategy._build_checkpointer = lambda *args, **kwargs: FakeCheckpointer()

    strategy.save_ckpt(_Wrapper(None), str(tmp_path), "global_step1", max_num=3, max_mem=0)
    strategy.save_ckpt(_Wrapper(None), str(tmp_path), "best_global_step2", max_num=3, max_mem=0)

    assert (tmp_path / "latest").read_text() == "global_step1"


def test_resolve_ckpt_load_dir_ignores_best_latest_when_regular_exists(tmp_path):
    strategy = object.__new__(FsdpStrategy)
    regular = tmp_path / "global_step1" / "model"
    best = tmp_path / "best_global_step2" / "model"
    regular.mkdir(parents=True)
    best.mkdir(parents=True)
    (tmp_path / "latest").write_text("best_global_step2")

    load_dir = strategy._resolve_ckpt_load_dir(str(tmp_path))

    assert load_dir == str(tmp_path / "global_step1")


def test_resolve_ckpt_load_dir_falls_back_from_stale_latest(tmp_path):
    strategy = object.__new__(FsdpStrategy)
    regular = tmp_path / "global_step3" / "model"
    regular.mkdir(parents=True)
    (tmp_path / "latest").write_text("global_step_missing")

    load_dir = strategy._resolve_ckpt_load_dir(str(tmp_path))

    assert load_dir == str(tmp_path / "global_step3")
