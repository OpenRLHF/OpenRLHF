import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "openrlhf/utils/math_utils.py"
_spec = importlib.util.spec_from_file_location("_math_unicode_test", _path)
math_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_utils)


@pytest.mark.parametrize(
    "answer, target, expected",
    [
        ("−2", "-2", True),
        ("−2", "-2.0", True),
        ("−1/2", "-1/2", True),
        ("1e−2", "0.01", True),
        ("2−x", "2-x", True),
        ("π", r"\pi", True),
        ("2π", r"2\pi", True),
        ("π/2", r"\pi/2", True),
        ("2×3", r"2\times 3", True),
        ("2·3", r"2\cdot 3", True),
        ("−2×π", r"-2\times\pi", True),
        ("(−2,π)", r"(-2,\pi)", True),
        ("−2", "2", False),
        ("−1/2", "1/2", False),
        ("1e−2", "100", False),
        ("x−1", "x+1", False),
        ("π", "3", False),
        ("2π", r"\pi", False),
        ("2×3", "2*4", False),
        ("2·3", "2+3", False),
    ],
)
def test_grade_boxed_unicode_symbols(answer, target, expected):
    extracted = math_utils.extract_boxed_answer(f"The answer is \\boxed{{{answer}}}.")
    assert math_utils.grade_answer(extracted, target) is expected
    assert math_utils.grade_answer(target, extracted) is expected
