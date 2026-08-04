import importlib.util
from pathlib import Path

import pytest


def _load_math_utils_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.math_utils", root / "openrlhf" / "utils" / "math_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract_boxed_answer = _load_math_utils_module().extract_boxed_answer


@pytest.mark.parametrize(
    ("solution", "expected"),
    [
        (r"answer: \boxed{42}", "42"),
        (r"answer: \fbox{42}", "42"),
        (r"answer: \fbox{\frac{1}{2}}", r"\frac{1}{2}"),
    ],
)
def test_extract_boxed_answer_supports_box_commands(solution, expected):
    assert extract_boxed_answer(solution) == expected


def test_extract_boxed_answer_rejects_unclosed_command():
    assert extract_boxed_answer(r"answer: \fbox{42") is None
