"""CPU regression tests for PEFT checkpoint saves without importing DeepSpeed."""

import ast
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class _Config:
    tie_word_embeddings = False

    def to_json_file(self, path):
        Path(path).write_text("{}")


class _PeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.saved = False

    def state_dict(self):
        return {
            "base_model.model.q_proj.lora_A.weight": torch.tensor([1.0]),
            "base_model.model.q_proj.base_layer.weight.nested_absmax": torch.tensor([2.0]),
        }

    def _consolidated_16bit_state_dict(self):
        return {"base_model.model.q_proj.lora_A.weight": torch.tensor([1.0])}

    def save_pretrained(self, output_dir, **kwargs):
        self.saved = True


class _PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()

    def state_dict(self):
        return {"q_proj.weight": torch.tensor([1.0]), "lm_head.weight": torch.tensor([2.0])}

    def _consolidated_16bit_state_dict(self):
        return {"lm_head.weight": torch.tensor([2.0])}

    def save_pretrained(self, output_dir, **kwargs):
        pass


@pytest.fixture
def save_model():
    source = Path(__file__).resolve().parents[1] / "openrlhf" / "utils" / "deepspeed" / "deepspeed.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    strategy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepspeedStrategy")
    method = next(node for node in strategy.body if isinstance(node, ast.FunctionDef) and node.name == "save_model")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {
        "os": os,
        "torch": torch,
        "nn": nn,
        "Actor": type("Actor", (), {}),
        "PeftModel": _PeftModel,
        "get_peft_model_state_dict": lambda model, state_dict: state_dict,
        "transformers": SimpleNamespace(AutoProcessor=None),
        "torch_dist_barrier_and_cuda_sync": lambda: None,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["save_model"]


def _strategy(zero_stage=3):
    strategy = SimpleNamespace()
    strategy.args = SimpleNamespace(ds=SimpleNamespace(zero_stage=zero_stage, tensor_parallel_size=1))
    strategy.ds_tensor_parallel_size = 1
    strategy.stage = zero_stage
    strategy.is_rank_0 = lambda: True
    strategy._unwrap_model = lambda model: model
    return strategy


def _tokenizer():
    return SimpleNamespace(save_pretrained=lambda *args, **kwargs: None)


def test_peft_save_ignores_missing_quantization_buffers(save_model):
    model = _PeftModel()

    with tempfile.TemporaryDirectory() as tmp:
        save_model(_strategy(), model, _tokenizer(), tmp)

        assert model.saved
        assert Path(tmp, "adapter_model.bin").exists()


def test_plain_model_mismatch_still_raises(save_model):
    model = _PlainModel()

    with tempfile.TemporaryDirectory() as tmp, pytest.raises(AssertionError, match="q_proj.weight"):
        save_model(_strategy(), model, _tokenizer(), tmp)


def test_plain_tied_embeddings_still_save(save_model):
    model = _PlainModel()
    model.config.tie_word_embeddings = True
    model._consolidated_16bit_state_dict = lambda: {"q_proj.weight": torch.tensor([1.0])}

    with tempfile.TemporaryDirectory() as tmp:
        save_model(_strategy(), model, _tokenizer(), tmp)
