import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_config_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("openrlhf.utils.config", root / "openrlhf" / "utils" / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


positive_int = _load_config_module().positive_int


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_positive_int_accepts_positive_value():
    assert positive_int("8") == 8
